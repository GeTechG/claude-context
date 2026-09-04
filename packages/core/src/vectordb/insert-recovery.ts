import { DroppedRow, InsertResult } from './types';

/**
 * #19 — one bad row must not cost the whole batch, and the loss must be visible.
 *
 * Before this, a batch that Milvus refused (`IllegalArgument (1100): length of
 * varchar field content exceeds max`) was thrown away wholesale: indexing
 * three.js lost five 100-chunk batches to two oversized files, and nine
 * unrelated files in those batches landed with zero chunks while the run still
 * printed `Re-indexing complete! 100%`.
 *
 * The recovery here is deliberately conservative about re-sending rows, because
 * Milvus `insert` does not de-duplicate on primary key — a re-sent row that was
 * already committed becomes a duplicate chunk that nothing will ever clean up:
 *
 *   - `err_index` non-empty means the request reached the server and the OTHER
 *     rows ARE written. That is checked first, whatever the status says: the
 *     rejected rows are recorded as dropped and nothing is re-sent;
 *   - a *whole-request* rejection (the server answered with a failure status)
 *     wrote nothing, so it is safe to retry without the rows that violate the
 *     schema, and failing that, row by row;
 *   - a THROWN error (a broken connection, a deadline) leaves it unknown
 *     whether the batch committed, so it is re-thrown, never retried. The
 *     caller counts the whole batch as lost — over-reporting the loss, which is
 *     the safe direction; a duplicate row is not recoverable, a missing one is;
 *   - if every row is rejected on its own, the problem is not the individual
 *     rows and the server's own message is thrown so the caller fails loudly.
 */

/**
 * How a single insert request came back. There is no 'threw' case on purpose:
 * a transport-level throw means "unknown whether committed" and is allowed to
 * propagate out of the whole recovery.
 */
type Attempt =
    | { kind: 'ok' }
    | { kind: 'partial'; errIndex: number[]; message: string }
    | { kind: 'rejected'; message: string };

export interface InsertRecoveryContext {
    collectionName: string;
    /** 'insert' | 'insertHybrid' — used verbatim in the log lines. */
    operation: string;
    rows: Record<string, any>[];
    /** Sends one insert request. Rejects or returns the SDK response. */
    insert: (rows: Record<string, any>[]) => Promise<any>;
    /**
     * Byte limits for varchar fields the caller cannot clamp (`content`,
     * `metadata`). Used to name the offending rows without a second round trip.
     */
    fieldLimits?: Record<string, number>;
    logger?: { warn: (...args: any[]) => void; error: (...args: any[]) => void };
}

function shorten(text: string, max = 200): string {
    const line = String(text).split('\n')[0];
    return line.length > max ? `${line.slice(0, max)}…` : line;
}

/** `path:start-end`, the form every dropped-row log line uses. */
export function describeRowLocation(row: Record<string, any>): string {
    const p = typeof row?.relativePath === 'string' && row.relativePath ? row.relativePath : '<unknown path>';
    const start = row?.startLine ?? 0;
    const end = row?.endLine ?? start;
    return `${p}:${start}-${end}`;
}

function toDropped(row: Record<string, any>, reason: string): DroppedRow {
    return {
        id: typeof row?.id === 'string' ? row.id : undefined,
        relativePath: typeof row?.relativePath === 'string' ? row.relativePath : undefined,
        startLine: typeof row?.startLine === 'number' ? row.startLine : undefined,
        endLine: typeof row?.endLine === 'number' ? row.endLine : undefined,
        contentBytes: typeof row?.content === 'string' ? Buffer.byteLength(row.content, 'utf8') : undefined,
        reason: shorten(reason),
    };
}

/** Fields that are over their declared byte limit, if any. */
export function fieldLimitViolations(
    row: Record<string, any>,
    fieldLimits: Record<string, number> = {},
): string[] {
    const violations: string[] = [];
    for (const [field, limit] of Object.entries(fieldLimits)) {
        const value = row?.[field];
        if (typeof value !== 'string') continue;
        const bytes = Buffer.byteLength(value, 'utf8');
        if (bytes > limit) violations.push(`${field} ${bytes}B > ${limit}B`);
    }
    return violations;
}

/**
 * Read a Milvus SDK insert response. The SDK reports schema/field validation
 * errors in the returned status object instead of throwing, so a batch that
 * "succeeded" here may have written nothing at all.
 */
export function readInsertResponse(resp: any, collectionName: string, operation: string, rowCount: number): Attempt {
    const status = resp?.status;
    // `err_index` FIRST: the request reached the server, so the rows outside
    // that list are committed. Re-sending them would duplicate primary keys,
    // whatever the status field happens to say.
    const errIndex: number[] = Array.isArray(resp?.err_index) ? resp.err_index.map(Number) : [];
    if (errIndex.length > 0) {
        return {
            kind: 'partial',
            errIndex,
            message:
                `[MilvusDB] ${operation} into '${collectionName}' partially failed: ` +
                `${errIndex.length}/${rowCount} rows rejected (status=${status?.reason || 'unknown'})`,
        };
    }
    // Either signal failing is a failure: a response carrying code 1100 with a
    // stale `error_code: 'Success'` (or the reverse) is not a successful insert.
    // An absent field says nothing and must not be read as failure.
    if (status) {
        const codeOk = status.code === undefined || status.code === null || status.code === 0;
        const nameOk = status.error_code === undefined || status.error_code === null || status.error_code === 'Success';
        if (!codeOk || !nameOk) {
            return {
                kind: 'rejected',
                message:
                    `[MilvusDB] ${operation} into '${collectionName}' failed: ` +
                    `${status.error_code || 'UnknownError'} (${status.code}): ${status.reason || 'no reason given'}`,
            };
        }
    }
    return { kind: 'ok' };
}

/**
 * Send one insert request. A throw is NOT caught: whether the batch committed
 * is unknown, and retrying an unknown insert is how duplicates are made.
 */
async function attempt(ctx: InsertRecoveryContext, rows: Record<string, any>[]): Promise<Attempt> {
    const resp = await ctx.insert(rows);
    return readInsertResponse(resp, ctx.collectionName, ctx.operation, rows.length);
}

/**
 * One line per dropped row (with its path), then one summary line.
 *
 * The summary line is what infra/lib/index-state.js counts as a dropped batch;
 * the per-row lines are excluded there by their "dropped row from" prefix,
 * because a row's `reason` can quote the server's own "N/M rows rejected" text
 * and would otherwise be counted as a batch of its own.
 */
function report(ctx: InsertRecoveryContext, dropped: DroppedRow[], total: number, inserted: number): void {
    const log = ctx.logger ?? console;
    for (const row of dropped) {
        log.error(
            `[MilvusDB] ❌ dropped row from ${ctx.operation} into '${ctx.collectionName}': ` +
            `${row.relativePath ?? '<unknown path>'}:${row.startLine ?? 0}-${row.endLine ?? 0} ` +
            `(content=${row.contentBytes ?? '?'}B, id=${row.id ?? '?'}): ${row.reason}`,
        );
    }
    // Keep this line parseable by infra/lib/index-state.js (it looks for
    // "<operation> into '<collection>'", "N/M rows rejected" and the
    // "dropped rows:" list, which it maps back to sources).
    log.error(
        `[MilvusDB] ❌ ${ctx.operation} into '${ctx.collectionName}' partially failed: ` +
        `${dropped.length}/${total} rows rejected, ${inserted} inserted after retry — dropped rows: ` +
        dropped.map((row) => `${row.relativePath ?? '<unknown path>'}:${row.startLine ?? 0}-${row.endLine ?? 0}`).join(', '),
    );
}

export async function insertRowsWithRecovery(ctx: InsertRecoveryContext): Promise<InsertResult> {
    const rows = ctx.rows;
    if (rows.length === 0) return { inserted: 0, dropped: [] };

    const first = await attempt(ctx, rows);
    if (first.kind === 'ok') return { inserted: rows.length, dropped: [] };

    // Partial rejection: the accepted rows are already written. Re-sending them
    // would duplicate primary keys, so only record what was lost.
    if (first.kind === 'partial') {
        const dropped = first.errIndex
            .filter((index) => index >= 0 && index < rows.length)
            .map((index) => toDropped(rows[index], describeReason(rows[index], ctx, first.message)));
        report(ctx, dropped, rows.length, rows.length - dropped.length);
        return { inserted: rows.length - dropped.length, dropped };
    }

    // Whole-request rejection: the server answered, and answered no, so nothing
    // was written. Retry without the rows we can name as schema violations (the
    // #19 case: `content` over 65535 bytes).
    const suspects = new Set<number>();
    rows.forEach((row, index) => {
        if (fieldLimitViolations(row, ctx.fieldLimits).length > 0) suspects.add(index);
    });

    if (suspects.size > 0 && suspects.size < rows.length) {
        const keep = rows.filter((_, index) => !suspects.has(index));
        const retry = await attempt(ctx, keep);
        if (retry.kind !== 'rejected') {
            const dropped = [...suspects]
                .sort((a, b) => a - b)
                .map((index) => toDropped(rows[index], describeReason(rows[index], ctx, first.message)));
            if (retry.kind === 'partial') {
                for (const index of retry.errIndex) {
                    if (index >= 0 && index < keep.length) dropped.push(toDropped(keep[index], retry.message));
                }
            }
            const inserted = keep.length - (retry.kind === 'partial' ? retry.errIndex.length : 0);
            report(ctx, dropped, rows.length, inserted);
            return { inserted, dropped };
        }
    }

    // Still refused (or nothing identifiable): insert row by row so one bad row
    // costs one row. Safe for the same reason as the retry above — every
    // attempt so far was answered with a rejection, so nothing is committed.
    const dropped: DroppedRow[] = [];
    let inserted = 0;
    for (const row of rows) {
        const single = await attempt(ctx, [row]);
        if (single.kind === 'ok') inserted++;
        else if (single.kind === 'partial') dropped.push(toDropped(row, describeReason(row, ctx, single.message)));
        else dropped.push(toDropped(row, describeReason(row, ctx, single.message)));
    }
    // Every row failing is not a row problem — surface the server's message.
    if (inserted === 0) throw new Error(first.message);
    if (dropped.length === 0) {
        // The batch failed as a whole but every row is individually fine (a
        // transient rejection). Nothing was lost, so this must NOT be reported
        // as a dropped batch.
        (ctx.logger ?? console).warn(
            `[MilvusDB] ${ctx.operation} into '${ctx.collectionName}': batch rejected but all ${inserted} rows inserted one by one`,
        );
        return { inserted, dropped };
    }
    report(ctx, dropped, rows.length, inserted);
    return { inserted, dropped };
}

function describeReason(row: Record<string, any>, ctx: InsertRecoveryContext, fallback: string): string {
    const violations = fieldLimitViolations(row, ctx.fieldLimits);
    return violations.length > 0 ? `schema limit exceeded: ${violations.join(', ')}` : fallback;
}
