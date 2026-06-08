import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";
import { Context, COLLECTION_LIMIT_MESSAGE, FileSynchronizer } from "@zilliz/claude-context-core";
import { SnapshotManager } from "./snapshot.js";
import type { CodebaseIndexOptions, RequestSplitterType } from "./config.js";
import { createRequestSplitter, isRequestSplitterType } from "./splitter.js";
import { ensureAbsolutePath, truncateContent, trackCodebasePath } from "./utils.js";
import {
    selectNeighbours,
    normalizeEdgeTypes,
    clampLimit,
    formatExpansion,
} from "./prose-graph-neighbours.js";
import {
    isUsageLogEnabled,
    newRequestId,
    newAnswerId,
    logRetrieval,
    logAnswer,
    relativise,
} from "./usage-logger.js";

export class ToolHandlers {
    private context: Context;
    private snapshotManager: SnapshotManager;
    private indexingStats: { indexedFiles: number; totalChunks: number } | null = null;
    private currentWorkspace: string;

    constructor(context: Context, snapshotManager: SnapshotManager) {
        this.context = context;
        this.snapshotManager = snapshotManager;
        this.currentWorkspace = process.cwd();
        console.log(`[WORKSPACE] Current workspace: ${this.currentWorkspace}`);
    }

    /**
     * Query Milvus for the real row count of a codebase's collection.
     * Returns null if the count cannot be determined — callers must NOT write a
     * snapshot entry in that case. Writing { indexedFiles: 0, totalChunks: 0,
     * status: 'completed' } for an unknown-state collection poisons the client:
     * the client treats 0/0 as "not indexed" and triggers force reindex, which
     * deletes real data and rewrites 0/0 — an infinite loop. See Issue #295.
     */
    private async queryCollectionStats(codebasePath: string): Promise<{ indexedFiles: number; totalChunks: number } | null> {
        try {
            const collectionName = this.context.getCollectionName(codebasePath);
            const rowCount = await this.context.getVectorDatabase().getCollectionRowCount(collectionName);
            if (rowCount < 0) {
                console.warn(`[SNAPSHOT-RECOVERY] Row count unknown for '${codebasePath}', skipping recovery write`);
                return null;
            }
            if (rowCount === 0) {
                console.warn(`[SNAPSHOT-RECOVERY] Collection '${collectionName}' truly empty — NOT writing recovered entry (would poison client)`);
                return null;
            }
            // rowCount is chunk count, not file count. Without a metadata query
            // we don't have the real file count; the snapshot will be corrected
            // on the next full index. Using rowCount for both is imprecise but
            // keeps the state non-zero so the client doesn't misread it as empty.
            return { indexedFiles: rowCount, totalChunks: rowCount };
        } catch (error) {
            console.warn(`[SNAPSHOT-RECOVERY] Failed to query stats for '${codebasePath}':`, error);
            return null;
        }
    }

    /**
     * One-shot startup validation: find any legacy 0/0+completed entries on disk
     * (left over from old MCP versions, v1 snapshot migrations, or pre-fix recovery
     * paths) and either heal them with the real Milvus row count or remove them
     * if the underlying collection is empty/missing. See Issue #295.
     *
     * Safe to call multiple times but intended to run once per server start after
     * loadCodebaseSnapshot(). Errors are caught and logged; never throws.
     */
    public async validateLegacyZeroEntries(): Promise<void> {
        try {
            const indexedCodebases = this.snapshotManager.getIndexedCodebases();
            let healed = 0, removed = 0, skipped = 0, checked = 0;

            for (const codebasePath of indexedCodebases) {
                const info = this.snapshotManager.getCodebaseInfo(codebasePath);
                if (!info || info.status !== 'indexed') continue;
                // Only validate suspiciously-zero entries
                if (info.indexedFiles !== 0 || info.totalChunks !== 0) continue;

                checked++;
                const collectionName = this.context.getCollectionName(codebasePath);
                const vdb = this.context.getVectorDatabase();

                // First probe: does the collection even exist? A "no" here is
                // authoritative (permanent orphan), while a throw is most likely
                // transient (Milvus unreachable) — keep those two cases distinct
                // so we don't destroy real state on a network blip.
                let collectionExists: boolean;
                try {
                    collectionExists = await vdb.hasCollection(collectionName);
                } catch (err) {
                    console.warn(`[SNAPSHOT-VALIDATE] hasCollection failed for '${codebasePath}' (likely transient), skipping:`, err);
                    skipped++;
                    continue;
                }

                if (!collectionExists) {
                    // Permanent orphan — no matching Milvus collection, so the
                    // 0/0+completed snapshot entry is a pure phantom. Remove it.
                    this.snapshotManager.removeCodebaseCompletely(codebasePath);
                    removed++;
                    console.warn(`[SNAPSHOT-VALIDATE] Removed orphan 0/0 entry '${codebasePath}' — no matching Milvus collection`);
                    continue;
                }

                // Collection exists — get an accurate row count.
                let rowCount: number;
                try {
                    rowCount = await vdb.getCollectionRowCount(collectionName);
                } catch (err) {
                    console.warn(`[SNAPSHOT-VALIDATE] getCollectionRowCount failed for '${codebasePath}', skipping:`, err);
                    skipped++;
                    continue;
                }

                if (rowCount > 0) {
                    // Heal: rewrite with real row count. rowCount is chunk count;
                    // without a cheap file-count query we reuse it for both fields.
                    // Imprecise but keeps the state non-zero and will be corrected
                    // on the next full index.
                    this.snapshotManager.setCodebaseIndexed(codebasePath, {
                        indexedFiles: rowCount,
                        totalChunks: rowCount,
                        status: 'completed' as const,
                    });
                    healed++;
                    console.log(`[SNAPSHOT-VALIDATE] Healed legacy 0/0 entry '${codebasePath}' → rows=${rowCount}`);
                } else if (rowCount === 0) {
                    // Collection exists but truly empty — the 0/0+completed entry
                    // is a phantom. Remove so the user must explicitly reindex.
                    this.snapshotManager.removeCodebaseCompletely(codebasePath);
                    removed++;
                    console.warn(`[SNAPSHOT-VALIDATE] Removed phantom 0/0 entry '${codebasePath}' — collection exists but empty`);
                } else {
                    // rowCount === -1 despite the collection existing: the count
                    // query failed after the existence probe succeeded. Treat as
                    // transient and leave the entry alone.
                    skipped++;
                    console.warn(`[SNAPSHOT-VALIDATE] Row count unavailable for existing collection '${codebasePath}', skipping`);
                }
            }

            if (healed > 0 || removed > 0) {
                this.snapshotManager.saveCodebaseSnapshot();
            }
            if (checked > 0) {
                console.log(`[SNAPSHOT-VALIDATE] Done — checked=${checked} healed=${healed} removed=${removed} skipped=${skipped}`);
            }
        } catch (error) {
            console.warn(`[SNAPSHOT-VALIDATE] Unexpected error during legacy 0/0 validation (non-fatal):`, error);
        }
    }

    /**
     * Sync indexed codebases from Zilliz Cloud collections
     * This method fetches all collections from the vector database,
     * extracts codebasePath from collection description (preferred) or falls back
     * to querying document metadata for old collections,
     * and updates the snapshot with discovered codebases.
     *
     * Logic: Compare mcp-codebase-snapshot.json with zilliz cloud collections
     * - If local snapshot has extra directories (not in cloud), remove them
     * - If local snapshot is missing directories (exist in cloud), ignore them
     */
    private async syncIndexedCodebasesFromCloud(): Promise<void> {
        try {
            console.log(`[SYNC-CLOUD] 🔄 Syncing indexed codebases from Zilliz Cloud...`);

            // Get all collections using the interface method
            const vectorDb = this.context.getVectorDatabase();

            // Use the new listCollections method from the interface
            const collections = await vectorDb.listCollections();

            console.log(`[SYNC-CLOUD] 📋 Found ${collections.length} collections in Zilliz Cloud`);

            if (collections.length === 0) {
                console.log(`[SYNC-CLOUD] ✅ No collections found in cloud. Skipping deletion of local codebases to avoid data loss from transient errors.`);
                return;
            }

            const cloudCodebases = new Set<string>();
            let codeCollectionsChecked = 0;
            let successfulExtractions = 0;

            // Check each collection for codebase path
            for (const collectionName of collections) {
                try {
                    // Skip collections that don't match the code_chunks pattern (support legacy, hybrid, and versioned hybrid_v* names)
                    if (!collectionName.startsWith('code_chunks_') && !collectionName.startsWith('hybrid_code_chunks_') && !/^hybrid_v[0-9a-z]+_code_chunks_/.test(collectionName)) {
                        console.log(`[SYNC-CLOUD] ⏭️  Skipping non-code collection: ${collectionName}`);
                        continue;
                    }

                    codeCollectionsChecked++;
                    console.log(`[SYNC-CLOUD] 🔍 Checking collection: ${collectionName}`);

                    // Try to extract codebasePath from collection description first (new format)
                    let extracted = false;
                    try {
                        const description = await vectorDb.getCollectionDescription(collectionName);
                        if (description && description.startsWith('codebasePath:')) {
                            const codebasePath = description.substring('codebasePath:'.length);
                            if (codebasePath.length > 0) {
                                console.log(`[SYNC-CLOUD] 📍 Found codebase path from description: ${codebasePath} in collection: ${collectionName}`);
                                cloudCodebases.add(codebasePath);
                                successfulExtractions++;
                                extracted = true;
                            }
                        }
                    } catch (descError: any) {
                        console.warn(`[SYNC-CLOUD] ⚠️  Failed to get description for collection ${collectionName}:`, descError.message || descError);
                    }

                    // Fallback: query document metadata for old collections without new description format
                    if (!extracted) {
                        console.log(`[SYNC-CLOUD] 🔄 Falling back to query-based extraction for collection: ${collectionName}`);
                        try {
                            const results = await vectorDb.query(
                                collectionName,
                                undefined as any, // Don't pass empty filter
                                ['metadata'], // Only fetch metadata field
                                1 // Only need one result to extract codebasePath
                            );

                            if (results && results.length > 0) {
                                const firstResult = results[0];
                                const metadataStr = firstResult.metadata;

                                if (metadataStr) {
                                    const metadata = JSON.parse(metadataStr);
                                    const codebasePath = metadata.codebasePath;

                                    if (codebasePath && typeof codebasePath === 'string') {
                                        console.log(`[SYNC-CLOUD] 📍 Found codebase path from query: ${codebasePath} in collection: ${collectionName}`);
                                        cloudCodebases.add(codebasePath);
                                        successfulExtractions++;
                                    } else {
                                        console.warn(`[SYNC-CLOUD] ⚠️  No codebasePath found in metadata for collection: ${collectionName}`);
                                    }
                                } else {
                                    console.warn(`[SYNC-CLOUD] ⚠️  No metadata found in collection: ${collectionName}`);
                                }
                            } else {
                                console.log(`[SYNC-CLOUD] ℹ️  Collection ${collectionName} is empty`);
                            }
                        } catch (queryError: any) {
                            console.warn(`[SYNC-CLOUD] ⚠️  Fallback query failed for collection ${collectionName}:`, queryError.message || queryError);
                        }
                    }
                } catch (collectionError: any) {
                    console.warn(`[SYNC-CLOUD] ⚠️  Error checking collection ${collectionName}:`, collectionError.message || collectionError);
                    // Continue with next collection
                }
            }

            console.log(`[SYNC-CLOUD] 📊 Found ${cloudCodebases.size} valid codebases in cloud (checked ${codeCollectionsChecked} code collections, ${successfulExtractions} successfully extracted)`);

            // Safety guard: if we checked code collections but none returned results,
            // treat this as an extraction failure rather than "cloud is empty".
            // This prevents deleting all local codebases due to transient errors.
            if (codeCollectionsChecked > 0 && successfulExtractions === 0) {
                console.warn(`[SYNC-CLOUD] ⚠️  All ${codeCollectionsChecked} code collection extractions failed. Skipping sync to avoid accidental deletion of local codebases.`);
                return;
            }

            // Get current local codebases
            const localCodebases = new Set(this.snapshotManager.getIndexedCodebases());
            console.log(`[SYNC-CLOUD] 📊 Found ${localCodebases.size} local codebases in snapshot`);

            let hasChanges = false;

            // Remove local codebases that don't exist in cloud
            for (const localCodebase of localCodebases) {
                if (!cloudCodebases.has(localCodebase)) {
                    this.snapshotManager.removeCodebaseCompletely(localCodebase);
                    hasChanges = true;

                    try {
                        await FileSynchronizer.deleteSnapshot(localCodebase);
                    } catch (error: any) {
                        console.warn(`[SYNC-CLOUD] ⚠️  Failed to delete local merkle snapshot for removed codebase '${localCodebase}':`, error?.message || error);
                    }

                    console.log(`[SYNC-CLOUD] ➖ Removed local codebase (not in cloud): ${localCodebase}`);
                }
            }

            // Add cloud codebases that are missing from local snapshot (recovery).
            // Query Milvus for the real row count — if unknown/empty, skip the write
            // so we don't persist a poisoning 0/0+completed entry (Issue #295).
            for (const cloudCodebase of cloudCodebases) {
                if (!localCodebases.has(cloudCodebase)) {
                    const stats = await this.queryCollectionStats(cloudCodebase);
                    if (stats) {
                        this.snapshotManager.setCodebaseIndexed(cloudCodebase, {
                            ...stats,
                            status: 'completed' as const
                        });
                        hasChanges = true;
                        console.log(`[SYNC-CLOUD] ➕ Recovered codebase from cloud: ${cloudCodebase} (rows=${stats.totalChunks})`);
                    } else {
                        console.log(`[SYNC-CLOUD] ⏭️  Skipped recovery for ${cloudCodebase} (row count unknown or zero)`);
                    }
                }
            }

            if (hasChanges) {
                this.snapshotManager.saveCodebaseSnapshot();
                console.log(`[SYNC-CLOUD] 💾 Updated snapshot to match cloud state`);
            } else {
                console.log(`[SYNC-CLOUD] ✅ Local snapshot already matches cloud state`);
            }

            console.log(`[SYNC-CLOUD] ✅ Cloud sync completed successfully`);
        } catch (error: any) {
            console.error(`[SYNC-CLOUD] ❌ Error syncing codebases from cloud:`, error.message || error);
            // Don't throw - this is not critical for the main functionality
        }
    }

    public async handleIndexCodebase(args: any) {
        const { path: codebasePath, force, splitter, customExtensions, ignorePatterns } = args;
        const forceReindex = force || false;
        const requestedSplitter = splitter || 'ast'; // Default to AST
        const customFileExtensions = customExtensions || [];
        const customIgnorePatterns = ignorePatterns || [];

        try {
            // Sync indexed codebases from cloud first
            await this.syncIndexedCodebasesFromCloud();

            // Validate splitter parameter
            if (!isRequestSplitterType(requestedSplitter)) {
                return {
                    content: [{
                        type: "text",
                        text: `Error: Invalid splitter type '${requestedSplitter}'. Must be 'ast' or 'langchain'.`
                    }],
                    isError: true
                };
            }
            const splitterType: RequestSplitterType = requestedSplitter;
            const indexOptions: CodebaseIndexOptions = {
                requestSplitter: splitterType,
                requestCustomExtensions: customFileExtensions,
                requestIgnorePatterns: customIgnorePatterns
            };
            // Force absolute path resolution - warn if relative path provided
            const absolutePath = ensureAbsolutePath(codebasePath);

            // Validate path exists
            if (!fs.existsSync(absolutePath)) {
                return {
                    content: [{
                        type: "text",
                        text: `Error: Path '${absolutePath}' does not exist. Original input: '${codebasePath}'`
                    }],
                    isError: true
                };
            }

            // Check if it's a directory
            const stat = fs.statSync(absolutePath);
            if (!stat.isDirectory()) {
                return {
                    content: [{
                        type: "text",
                        text: `Error: Path '${absolutePath}' is not a directory`
                    }],
                    isError: true
                };
            }

            // Check if already indexing
            if (this.snapshotManager.getIndexingCodebases().includes(absolutePath)) {
                if (forceReindex) {
                    console.log(`[FORCE-REINDEX] Clearing stale indexing state for '${absolutePath}'`);
                    this.snapshotManager.removeCodebaseCompletely(absolutePath);
                    this.snapshotManager.saveCodebaseSnapshot();
                } else {
                    return {
                        content: [{
                            type: "text",
                            text: `Codebase '${absolutePath}' is already being indexed in the background. Please wait for completion.`
                        }],
                        isError: true
                    };
                }
            }

            //Check if the snapshot and cloud index are in sync
            const snapshotHasIndex = this.snapshotManager.getIndexedCodebases().includes(absolutePath);
            const vectorDbHasIndex = await this.context.hasIndex(absolutePath);
            if (snapshotHasIndex !== vectorDbHasIndex) {
                if (vectorDbHasIndex && !snapshotHasIndex) {
                    // Query Milvus for real row count. If unknown/empty, log and move on
                    // without writing 0/0+completed (which would trigger the force-reindex
                    // loop in Issue #295). The user is about to (re)index anyway.
                    const stats = await this.queryCollectionStats(absolutePath);
                    if (stats) {
                        console.warn(`[INDEX-VALIDATION] Recovering missing snapshot for '${absolutePath}' (rows=${stats.totalChunks})`);
                        this.snapshotManager.setCodebaseIndexed(absolutePath, { ...stats, status: 'completed' as const });
                        this.snapshotManager.saveCodebaseSnapshot();
                    } else {
                        console.warn(`[INDEX-VALIDATION] VectorDB reports index for '${absolutePath}' but row count unknown/zero — not writing snapshot entry`);
                    }
                } else if (!vectorDbHasIndex && snapshotHasIndex) {
                    console.warn(`[INDEX-VALIDATION] Clearing stale snapshot for '${absolutePath}'`);
                    this.snapshotManager.removeCodebaseCompletely(absolutePath);
                    this.snapshotManager.saveCodebaseSnapshot();
                }
            }

            // Check if already indexed (unless force is true)
            if (!forceReindex && this.snapshotManager.getIndexedCodebases().includes(absolutePath)) {
                return {
                    content: [{
                        type: "text",
                        text: `Codebase '${absolutePath}' is already indexed. Use force=true to re-index.`
                    }],
                    isError: true
                };
            }

            // If force reindex and codebase is already indexed, remove it
            if (forceReindex) {
                this.snapshotManager.removeCodebaseCompletely(absolutePath);
                this.snapshotManager.saveCodebaseSnapshot();
                if (await this.context.hasIndex(absolutePath)) {
                    console.log(`[FORCE-REINDEX] 🔄 Clearing index for '${absolutePath}'`);
                    await this.context.clearIndex(absolutePath);
                }
            }

            // CRITICAL: Pre-index collection creation validation
            try {
                console.log(`[INDEX-VALIDATION] 🔍 Validating collection creation capability`);
                const canCreateCollection = await this.context.getVectorDatabase().checkCollectionLimit();

                if (!canCreateCollection) {
                    console.error(`[INDEX-VALIDATION] ❌ Collection limit validation failed: ${absolutePath}`);

                    // CRITICAL: Immediately return the COLLECTION_LIMIT_MESSAGE to MCP client
                    return {
                        content: [{
                            type: "text",
                            text: COLLECTION_LIMIT_MESSAGE
                        }],
                        isError: true
                    };
                }

                console.log(`[INDEX-VALIDATION] ✅  Collection creation validation completed`);
            } catch (validationError: any) {
                // Handle other collection creation errors
                console.error(`[INDEX-VALIDATION] ❌ Collection creation validation failed:`, validationError);
                return {
                    content: [{
                        type: "text",
                        text: `Error validating collection creation: ${validationError.message || validationError}`
                    }],
                    isError: true
                };
            }

            if (customFileExtensions.length > 0) {
                console.log(`[CUSTOM-EXTENSIONS] Using ${customFileExtensions.length} request-scoped custom extensions: ${customFileExtensions.join(', ')}`);
            }

            // Check current status and log if retrying after failure
            const currentStatus = this.snapshotManager.getCodebaseStatus(absolutePath);
            if (currentStatus === 'indexfailed') {
                const failedInfo = this.snapshotManager.getCodebaseInfo(absolutePath) as any;
                console.log(`[BACKGROUND-INDEX] Retrying indexing for previously failed codebase. Previous error: ${failedInfo?.errorMessage || 'Unknown error'}`);
            }

            // Set to indexing status and save snapshot immediately
            this.snapshotManager.setCodebaseIndexing(absolutePath, 0, indexOptions);
            this.snapshotManager.saveCodebaseSnapshot();

            // Track the codebase path for syncing
            trackCodebasePath(absolutePath);

            // Start background indexing - now safe to proceed
            this.startBackgroundIndexing(absolutePath, forceReindex, splitterType, customIgnorePatterns, customFileExtensions, indexOptions);

            const pathInfo = codebasePath !== absolutePath
                ? `\nNote: Input path '${codebasePath}' was resolved to absolute path '${absolutePath}'`
                : '';

            const extensionInfo = customFileExtensions.length > 0
                ? `\nUsing ${customFileExtensions.length} custom extensions: ${customFileExtensions.join(', ')}`
                : '';

            const ignoreInfo = customIgnorePatterns.length > 0
                ? `\nUsing ${customIgnorePatterns.length} custom ignore patterns: ${customIgnorePatterns.join(', ')}`
                : '';

            return {
                content: [{
                    type: "text",
                    text: `Started background indexing for codebase '${absolutePath}' using ${splitterType.toUpperCase()} splitter.${pathInfo}${extensionInfo}${ignoreInfo}\n\nIndexing is running in the background. You can search the codebase while indexing is in progress, but results may be incomplete until indexing completes.`
                }]
            };

        } catch (error: any) {
            // Enhanced error handling to prevent MCP service crash
            console.error('Error in handleIndexCodebase:', error);

            // Ensure we always return a proper MCP response, never throw
            return {
                content: [{
                    type: "text",
                    text: `Error starting indexing: ${error.message || error}`
                }],
                isError: true
            };
        }
    }

    private async startBackgroundIndexing(
        codebasePath: string,
        forceReindex: boolean,
        splitterType: RequestSplitterType,
        customIgnorePatterns: string[] = [],
        customFileExtensions: string[] = [],
        indexOptions?: CodebaseIndexOptions
    ) {
        const absolutePath = codebasePath;
        let lastSaveTime = 0; // Track last save timestamp

        try {
            console.log(`[BACKGROUND-INDEX] Starting background indexing for: ${absolutePath}`);

            // Note: If force reindex, collection was already cleared during validation phase
            if (forceReindex) {
                console.log(`[BACKGROUND-INDEX] ℹ️  Force reindex mode - collection was already cleared during validation`);
            }

            const requestSplitter = createRequestSplitter(splitterType);

            // Load ignore patterns from files first (including .ignore, .gitignore, etc.)
            // and merge them with this request's custom ignore patterns without
            // relying on shared Context state for this background indexing task.
            const ignorePatterns = await this.context.getEffectiveIgnorePatterns(absolutePath, customIgnorePatterns);
            const supportedExtensions = this.context.getEffectiveSupportedExtensions(customFileExtensions);

            // Initialize file synchronizer with proper ignore patterns (including project-specific patterns)
            console.log(`[BACKGROUND-INDEX] Using ignore patterns: ${ignorePatterns.join(', ')}`);
            if (customFileExtensions.length > 0) {
                console.log(`[BACKGROUND-INDEX] Using ${customFileExtensions.length} request-scoped custom extensions: ${customFileExtensions.join(', ')}`);
            }
            const synchronizer = new FileSynchronizer(absolutePath, ignorePatterns, supportedExtensions);
            await synchronizer.initialize();

            // Store synchronizer in the context (let context manage collection names)
            await this.context.getPreparedCollection(absolutePath);
            const collectionName = this.context.getCollectionName(absolutePath);
            this.context.setSynchronizer(collectionName, synchronizer);

            console.log(`[BACKGROUND-INDEX] Starting indexing with ${splitterType} splitter for: ${absolutePath}`);

            // Log embedding provider information before indexing
            const embeddingProvider = this.context.getEmbedding();
            console.log(`[BACKGROUND-INDEX] 🧠 Using embedding provider: ${embeddingProvider.getProvider()} with dimension: ${embeddingProvider.getDimension()}`);

            // Start indexing with the appropriate context and progress tracking
            console.log(`[BACKGROUND-INDEX] 🚀 Beginning codebase indexing process...`);
            const stats = await this.context.indexCodebase(absolutePath, (progress) => {
                // Update progress in snapshot manager using new method
                this.snapshotManager.setCodebaseIndexing(absolutePath, progress.percentage);

                // Save snapshot periodically (every 2 seconds to avoid too frequent saves)
                const currentTime = Date.now();
                if (currentTime - lastSaveTime >= 2000) { // 2 seconds = 2000ms
                    this.snapshotManager.saveCodebaseSnapshot();
                    lastSaveTime = currentTime;
                    console.log(`[BACKGROUND-INDEX] 💾 Saved progress snapshot at ${progress.percentage.toFixed(1)}%`);
                }

                console.log(`[BACKGROUND-INDEX] Progress: ${progress.phase} - ${progress.percentage}% (${progress.current}/${progress.total})`);
            }, false, customIgnorePatterns, customFileExtensions, requestSplitter);
            console.log(`[BACKGROUND-INDEX] ✅ Indexing completed successfully! Files: ${stats.indexedFiles}, Chunks: ${stats.totalChunks}`);

            // Set codebase to indexed status with complete statistics
            this.snapshotManager.setCodebaseIndexed(absolutePath, stats, indexOptions);
            this.indexingStats = { indexedFiles: stats.indexedFiles, totalChunks: stats.totalChunks };

            // Save snapshot after updating codebase lists
            this.snapshotManager.saveCodebaseSnapshot();

            let message = `Background indexing completed for '${absolutePath}' using ${splitterType.toUpperCase()} splitter.\nIndexed ${stats.indexedFiles} files, ${stats.totalChunks} chunks.`;
            if (stats.status === 'limit_reached') {
                message += `\n⚠️  Warning: Indexing stopped because the chunk limit (450,000) was reached. The index may be incomplete.`;
            }

            console.log(`[BACKGROUND-INDEX] ${message}`);

        } catch (error: any) {
            console.error(`[BACKGROUND-INDEX] Error during indexing for ${absolutePath}:`, error);

            // Get the last attempted progress
            const lastProgress = this.snapshotManager.getIndexingProgress(absolutePath);

            // Set codebase to failed status with error information
            const errorMessage = error.message || String(error);
            this.snapshotManager.setCodebaseIndexFailed(absolutePath, errorMessage, lastProgress, indexOptions);
            this.snapshotManager.saveCodebaseSnapshot();

            // Log error but don't crash MCP service - indexing errors are handled gracefully
            console.error(`[BACKGROUND-INDEX] Indexing failed for ${absolutePath}: ${errorMessage}`);
        }
    }

    public async handleSearchCode(args: any) {
        const { path: codebasePath, query, limit = 10, extensionFilter, shapeHint } = args;
        const resultLimit = limit || 10;

        // agentic-reference-context-assembler: optional query_shape hint.
        // Validate against the router's QueryShape set; any other value
        // (or absence) degrades to undefined → semanticSearch is called
        // without a 6th arg, byte-identical to the pre-change one-shot path
        // (concept-span quota stays a no-op). Never throws on bad input.
        const VALID_SHAPE_HINTS = ['single', 'multi-hop', 'comparison', 'concept'];
        const queryShape: string | undefined =
            typeof shapeHint === 'string' && VALID_SHAPE_HINTS.includes(shapeHint)
                ? shapeHint
                : undefined;

        try {
            // Sync indexed codebases from cloud first
            await this.syncIndexedCodebasesFromCloud();

            // Force absolute path resolution - warn if relative path provided
            const absolutePath = ensureAbsolutePath(codebasePath);

            // Validate path exists
            if (!fs.existsSync(absolutePath)) {
                return {
                    content: [{
                        type: "text",
                        text: `Error: Path '${absolutePath}' does not exist. Original input: '${codebasePath}'`
                    }],
                    isError: true
                };
            }

            // Check if it's a directory
            const stat = fs.statSync(absolutePath);
            if (!stat.isDirectory()) {
                return {
                    content: [{
                        type: "text",
                        text: `Error: Path '${absolutePath}' is not a directory`
                    }],
                    isError: true
                };
            }

            trackCodebasePath(absolutePath);

            // Check if this codebase is indexed or being indexed
            const indexedCodebasePath = this.snapshotManager.findIndexedCodebasePath(absolutePath);
            const indexingCodebasePath = this.snapshotManager.findIndexingCodebasePath(absolutePath);
            const matchedCodebase = [indexedCodebasePath, indexingCodebasePath]
                .filter((codebase): codebase is string => codebase !== undefined)
                .sort((a, b) => b.length - a.length)[0];
            let searchCodebasePath = matchedCodebase || absolutePath;
            let isIndexed = indexedCodebasePath === searchCodebasePath;
            const isIndexing = indexingCodebasePath === searchCodebasePath;

            if (!isIndexed && !isIndexing) {
                // Fallback: check VectorDB directly in case snapshot is out of sync.
                // Only recover the snapshot when we can confirm a real row count —
                // writing 0/0+completed for an unverifiable collection poisons the
                // client into a force-reindex loop (Issue #295).
                const hasVectorIndex = await this.context.hasIndex(absolutePath);
                if (hasVectorIndex) {
                    const stats = await this.queryCollectionStats(absolutePath);
                    if (stats) {
                        console.warn(`[SEARCH] Snapshot missing but VectorDB has index for '${absolutePath}', recovering snapshot (rows=${stats.totalChunks})`);
                        this.snapshotManager.setCodebaseIndexed(absolutePath, { ...stats, status: 'completed' as const });
                        this.snapshotManager.saveCodebaseSnapshot();
                        searchCodebasePath = absolutePath;
                        isIndexed = true;
                        // Continue with search (don't return error)
                    } else {
                        return {
                            content: [{
                                type: "text",
                                text: `Error: Codebase '${absolutePath}' is not indexed. Please index it first using the index_codebase tool.`
                            }],
                            isError: true
                        };
                    }
                } else {
                    return {
                        content: [{
                            type: "text",
                            text: `Error: Codebase '${absolutePath}' is not indexed. Please index it first using the index_codebase tool.`
                        }],
                        isError: true
                    };
                }
            }

            // Show indexing status if codebase is being indexed
            let indexingStatusMessage = '';
            if (isIndexing) {
                indexingStatusMessage = `\n⚠️  **Indexing in Progress**: This codebase is currently being indexed in the background. Search results may be incomplete until indexing completes.`;
            }

            console.log(`[SEARCH] Searching in codebase: ${searchCodebasePath}`);
            console.log(`[SEARCH] Query: "${query}"`);
            console.log(`[SEARCH] Indexing status: ${isIndexing ? 'In Progress' : 'Completed'}`);

            // Log embedding provider information before search
            const embeddingProvider = this.context.getEmbedding();
            console.log(`[SEARCH] 🧠 Using embedding provider: ${embeddingProvider.getProvider()} for search`);
            console.log(`[SEARCH] 🔍 Generating embeddings for query using ${embeddingProvider.getProvider()}...`);

            // Build filter expression from extensionFilter list
            let filterExpr: string | undefined = undefined;
            if (Array.isArray(extensionFilter) && extensionFilter.length > 0) {
                const cleaned = extensionFilter
                    .filter((v: any) => typeof v === 'string')
                    .map((v: string) => v.trim())
                    .filter((v: string) => v.length > 0);
                const invalid = cleaned.filter((e: string) => !(e.startsWith('.') && e.length > 1 && !/\s/.test(e)));
                if (invalid.length > 0) {
                    return {
                        content: [{ type: 'text', text: `Error: Invalid file extensions in extensionFilter: ${JSON.stringify(invalid)}. Use proper extensions like '.ts', '.py'.` }],
                        isError: true
                    };
                }
                const quoted = cleaned.map((e: string) => `'${e}'`).join(', ');
                filterExpr = `fileExtension in [${quoted}]`;
            }

            // Search in the specified codebase
            const searchResults = await this.context.semanticSearch(
                searchCodebasePath,
                query,
                Math.min(resultLimit, 50),
                0.3,
                filterExpr,
                queryShape
            );

            console.log(`[SEARCH] ✅ Search completed! Found ${searchResults.length} results using ${embeddingProvider.getProvider()} embeddings`);

            if (searchResults.length === 0) {
                // Check if collection was lost (indexed locally but missing in Milvus)
                if (isIndexed && !isIndexing) {
                    const collectionName = this.context.getCollectionName(searchCodebasePath);
                    const hasCollection = await this.context.getVectorDatabase().hasCollection(collectionName);
                    if (!hasCollection) {
                        return {
                            content: [{ type: "text", text: `Error: Index data for '${searchCodebasePath}' has been lost (collection not found in Milvus). Please re-index using index_codebase with force=true.` }],
                            isError: true
                        };
                    }
                }

                let noResultsMessage = `No results found for query: "${query}" in codebase '${searchCodebasePath}'`;
                if (searchCodebasePath !== absolutePath) {
                    noResultsMessage += `\nRequested path '${absolutePath}' is covered by indexed codebase '${searchCodebasePath}'.`;
                }
                if (isIndexing) {
                    noResultsMessage += `\n\nNote: This codebase is still being indexed. Try searching again after indexing completes, or the query may not match any indexed content.`;
                }
                return {
                    content: [{
                        type: "text",
                        text: noResultsMessage
                    }]
                };
            }

            // Format results
            const formattedResults = searchResults.map((result: any, index: number) => {
                const location = `${result.relativePath}:${result.startLine}-${result.endLine}`;
                const context = truncateContent(result.content, 5000);
                const codebaseInfo = path.basename(searchCodebasePath);

                // prose-graph-mcp-tool: surface the stable chunk_id as the
                // agent's handle for `expand_context` graph navigation.
                // Additive + backward-compatible: omitted for legacy chunks
                // that predate chunk_id exposure; all other lines byte-stable.
                const chunkIdLine = result.chunk_id ? `   Chunk-ID: ${result.chunk_id}\n` : '';

                return `${index + 1}. Code snippet (${result.language}) [${codebaseInfo}]\n` +
                    `   Location: ${location}\n` +
                    chunkIdLine +
                    `   Rank: ${index + 1}\n` +
                    `   Context: \n\`\`\`${result.language}\n${context}\n\`\`\`\n`;
            }).join('\n');

            let resultMessage = `Found ${searchResults.length} results for query: "${query}" in codebase '${searchCodebasePath}'${indexingStatusMessage}`;
            if (searchCodebasePath !== absolutePath) {
                resultMessage += `\nRequested path '${absolutePath}' is covered by indexed codebase '${searchCodebasePath}'.`;
            }
            resultMessage += `\n\n${formattedResults}`;

            // Phase 3: surface candidate symbols extracted from the result
            // pool so the agent can hand them to Serena's find_symbol for
            // full-definition lookup (see local-rag SKILL.md).
            const candidateSymbols = (searchResults[0] as any)?.candidateSymbols;
            if (Array.isArray(candidateSymbols) && candidateSymbols.length > 0) {
                resultMessage += `\n\nCandidate symbols (top ${candidateSymbols.length}, for Serena \`find_symbol\` enrichment): ${candidateSymbols.join(', ')}`;
            }

            if (isIndexing) {
                resultMessage += `\n\n💡 **Tip**: This codebase is still being indexed. More results may become available as indexing progresses.`;
            }

            // usage-logging-dataset: capture this real retrieval call (query +
            // full returned context) and surface a Request-ID so a later
            // record_answer can correlate the answer back to it. Additive,
            // best-effort, and fully guarded — when logging is disabled the
            // response is byte-identical to the pre-change format and no work is
            // done; a logging throw can never reach the tool result.
            if (isUsageLogEnabled()) {
                try {
                    const requestId = newRequestId();
                    resultMessage = `Request-ID: ${requestId}\n${resultMessage}`;
                    logRetrieval({
                        request_id: requestId,
                        tool: "search_code",
                        codebase: relativise(searchCodebasePath),
                        query,
                        ...(filterExpr ? { extensionFilter } : {}),
                        ...(queryShape ? { shapeHint: queryShape } : {}),
                        result_count: searchResults.length,
                        results: searchResults.map((r: any, i: number) => ({
                            rank: i + 1,
                            score: r.score,
                            chunk_id: r.chunk_id,
                            relativePath: relativise(r.relativePath),
                            startLine: r.startLine,
                            endLine: r.endLine,
                            content: truncateContent(r.content ?? "", 5000),
                        })),
                    });
                } catch (logErr: any) {
                    console.error(`[USAGE-LOG] search_code capture failed: ${logErr?.message ?? logErr}`);
                }
            }

            return {
                content: [{
                    type: "text",
                    text: resultMessage
                }]
            };
        } catch (error) {
            // Check if this is the collection limit error
            // Handle both direct string throws and Error objects containing the message
            const errorMessage = typeof error === 'string' ? error : (error instanceof Error ? error.message : String(error));

            if (errorMessage === COLLECTION_LIMIT_MESSAGE || errorMessage.includes(COLLECTION_LIMIT_MESSAGE)) {
                // Return the collection limit message as a successful response
                // This ensures LLM treats it as final answer, not as retryable error
                return {
                    content: [{
                        type: "text",
                        text: COLLECTION_LIMIT_MESSAGE
                    }]
                };
            }

            return {
                content: [{
                    type: "text",
                    text: `Error searching code: ${errorMessage} Please check if the codebase has been indexed first.`
                }],
                isError: true
            };
        }
    }

    /**
     * prose-graph-mcp-tool: agent-callable 1-hop traversal over the
     * deterministic prose-graph side-index (`<root>/.prose-graph.json`).
     * Given a `chunk_id` from a prior `search_code` result (its `Chunk-ID`
     * line), return that chunk's neighbours with their relationship type,
     * weight, and (collection-aware) content. Pure in-memory graph lookup +
     * one Milvus fetch — no LLM, no GPU, no ranking change. Degrades
     * gracefully when the side-index is absent.
     */
    public async handleExpandContext(args: any) {
        const { path: codebasePath, chunk_id, edge_types, limit } = args ?? {};

        if (typeof chunk_id !== "string" || chunk_id.trim().length === 0) {
            return {
                content: [{
                    type: "text",
                    text: "Error: expand_context requires a non-empty `chunk_id` (the `Chunk-ID` line from a search_code result)."
                }],
                isError: true
            };
        }

        try {
            // Sync indexed codebases from cloud first (mirrors handleSearchCode).
            await this.syncIndexedCodebasesFromCloud();

            const absolutePath = ensureAbsolutePath(codebasePath);
            if (!fs.existsSync(absolutePath)) {
                return {
                    content: [{
                        type: "text",
                        text: `Error: Path '${absolutePath}' does not exist. Original input: '${codebasePath}'`
                    }],
                    isError: true
                };
            }

            // Resolve to the indexed codebase root (mirrors handleSearchCode) so
            // the prose-graph side-index path and the prose collection both match
            // what search_code used.
            const indexedCodebasePath = this.snapshotManager.findIndexedCodebasePath(absolutePath);
            const indexingCodebasePath = this.snapshotManager.findIndexingCodebasePath(absolutePath);
            const matchedCodebase = [indexedCodebasePath, indexingCodebasePath]
                .filter((codebase): codebase is string => codebase !== undefined)
                .sort((a, b) => b.length - a.length)[0];
            const searchCodebasePath = matchedCodebase || absolutePath;

            // Graceful-off: missing / unparseable / version-mismatch side-index.
            const index = this.context.getProseGraphIndex(searchCodebasePath);
            if (!index) {
                return {
                    content: [{
                        type: "text",
                        text: `Prose-graph index not available for '${searchCodebasePath}' (missing, unparseable, or version-mismatched). Build it with infra/build-prose-graph.js. search_code and all other tools are unaffected.`
                    }]
                };
            }

            const seedId = chunk_id.trim();
            const neighbours = index.neighbours(seedId);
            if (!neighbours || neighbours.length === 0) {
                return {
                    content: [{
                        type: "text",
                        text: `No graph neighbours for \`${seedId}\` (chunk absent from the prose-graph, or it has no 1-hop edges).`
                    }]
                };
            }

            const edgeFilter = normalizeEdgeTypes(edge_types);
            const cap = clampLimit(limit);
            const selected = selectNeighbours(neighbours, edgeFilter, cap);

            // Distinct neighbour count after the edge_types filter, for the header.
            const totalAfterFilter = (() => {
                const ids = new Set<string>();
                for (const e of neighbours) {
                    if (edgeFilter && !edgeFilter.has(e.type)) continue;
                    if (e.to) ids.add(e.to);
                }
                return ids.size;
            })();

            // Batch-fetch neighbour content from the prose collection; stale ids
            // (absent from Milvus) silently drop out and are noted by the formatter.
            const neighbourIds = selected.map((e) => e.to);
            const fetched = await this.context.fetchProseChunksByIds(searchCodebasePath, neighbourIds);
            const contentById = new Map<string, any>();
            for (const r of fetched) {
                if (r.chunk_id) contentById.set(r.chunk_id, r);
            }

            let text = formatExpansion(seedId, selected, contentById, totalAfterFilter);

            // usage-logging-dataset: capture this real expand_context call (seed
            // + returned neighbours) and surface a Request-ID, mirroring
            // handleSearchCode. Additive, best-effort, fully guarded.
            if (isUsageLogEnabled()) {
                try {
                    const requestId = newRequestId();
                    text = `Request-ID: ${requestId}\n${text}`;
                    logRetrieval({
                        request_id: requestId,
                        tool: "expand_context",
                        codebase: relativise(searchCodebasePath),
                        seed_chunk_id: seedId,
                        ...(edgeFilter ? { edge_types: Array.from(edgeFilter) } : {}),
                        ...(limit !== undefined ? { limit: cap } : {}),
                        result_count: selected.length,
                        neighbours: selected.map((e) => {
                            const r = contentById.get(e.to);
                            return {
                                chunk_id: e.to,
                                relationship: e.type,
                                weight: e.weight,
                                relativePath: r ? relativise(r.relativePath) : undefined,
                                startLine: r?.startLine,
                                endLine: r?.endLine,
                                content: r ? truncateContent(r.content ?? "", 5000) : undefined,
                            };
                        }),
                    });
                } catch (logErr: any) {
                    console.error(`[USAGE-LOG] expand_context capture failed: ${logErr?.message ?? logErr}`);
                }
            }

            return { content: [{ type: "text", text }] };
        } catch (error) {
            const errorMessage = typeof error === "string"
                ? error
                : (error instanceof Error ? error.message : String(error));
            return {
                content: [{
                    type: "text",
                    text: `Error expanding context: ${errorMessage}`
                }],
                isError: true
            };
        }
    }

    /**
     * usage-logging-dataset: capture the final synthesized answer that the MCP
     * server cannot observe (it is produced by the consuming agent out of the
     * retrieved context). The agent calls this at the end of a reference-answer
     * task with the `answer` text, the `request_ids` it drew on (from the
     * `Request-ID` lines of its search_code / expand_context calls), and an
     * optional in-session quality signal. Best-effort and default-safe: a
     * logging failure never errors, and when capture is disabled it returns a
     * benign notice instead.
     */
    public async handleRecordAnswer(args: any) {
        const { answer, request_ids, signal, note, query } = args ?? {};

        if (typeof answer !== "string" || answer.trim().length === 0) {
            return {
                content: [{
                    type: "text",
                    text: "Error: record_answer requires a non-empty `answer` string."
                }],
                isError: true
            };
        }

        const requestIds = Array.isArray(request_ids)
            ? request_ids.filter((id: any) => typeof id === "string" && id.trim().length > 0)
            : [];
        if (requestIds.length === 0) {
            return {
                content: [{
                    type: "text",
                    text: "Error: record_answer requires `request_ids` — a non-empty array of the Request-ID(s) returned by the search_code / expand_context calls this answer used."
                }],
                isError: true
            };
        }

        if (!isUsageLogEnabled()) {
            return {
                content: [{
                    type: "text",
                    text: "Usage logging is disabled (RAG_USAGE_LOG); answer not recorded."
                }]
            };
        }

        const VALID_SIGNALS = ["helpful", "not_helpful", "unknown"];
        const normalizedSignal = typeof signal === "string" && VALID_SIGNALS.includes(signal)
            ? signal
            : undefined;

        try {
            const recordId = newAnswerId();
            logAnswer({
                record_id: recordId,
                answer,
                request_ids: requestIds,
                ...(normalizedSignal ? { signal: normalizedSignal } : {}),
                ...(typeof note === "string" && note.length > 0 ? { note } : {}),
                ...(typeof query === "string" && query.length > 0 ? { query } : {}),
            });
            return {
                content: [{
                    type: "text",
                    text: `Recorded answer ${recordId} against ${requestIds.length} request id(s).`
                }]
            };
        } catch (error: any) {
            // Best-effort: a logging failure must not surface as a tool error.
            console.error(`[USAGE-LOG] record_answer capture failed: ${error?.message ?? error}`);
            return {
                content: [{
                    type: "text",
                    text: "Answer capture encountered an error and was skipped (logged to stderr)."
                }]
            };
        }
    }

    public async handleClearIndex(args: any) {
        const { path: codebasePath } = args;

        if (this.snapshotManager.getIndexedCodebases().length === 0 && this.snapshotManager.getIndexingCodebases().length === 0) {
            return {
                content: [{
                    type: "text",
                    text: "No codebases are currently indexed or being indexed."
                }]
            };
        }

        try {
            // Force absolute path resolution - warn if relative path provided
            const absolutePath = ensureAbsolutePath(codebasePath);

            // Validate path exists
            if (!fs.existsSync(absolutePath)) {
                return {
                    content: [{
                        type: "text",
                        text: `Error: Path '${absolutePath}' does not exist. Original input: '${codebasePath}'`
                    }],
                    isError: true
                };
            }

            // Check if it's a directory
            const stat = fs.statSync(absolutePath);
            if (!stat.isDirectory()) {
                return {
                    content: [{
                        type: "text",
                        text: `Error: Path '${absolutePath}' is not a directory`
                    }],
                    isError: true
                };
            }

            // Check if this codebase is indexed or being indexed
            const isIndexed = this.snapshotManager.getIndexedCodebases().includes(absolutePath);
            const isIndexing = this.snapshotManager.getIndexingCodebases().includes(absolutePath);

            if (!isIndexed && !isIndexing) {
                return {
                    content: [{
                        type: "text",
                        text: `Error: Codebase '${absolutePath}' is not indexed or being indexed.`
                    }],
                    isError: true
                };
            }

            console.log(`[CLEAR] Clearing codebase: ${absolutePath}`);

            try {
                await this.context.clearIndex(absolutePath);
                console.log(`[CLEAR] Successfully cleared index for: ${absolutePath}`);
            } catch (error: any) {
                const errorMsg = `Failed to clear ${absolutePath}: ${error.message}`;
                console.error(`[CLEAR] ${errorMsg}`);
                return {
                    content: [{
                        type: "text",
                        text: errorMsg
                    }],
                    isError: true
                };
            }

            // Completely remove the cleared codebase from snapshot
            this.snapshotManager.removeCodebaseCompletely(absolutePath);

            // Reset indexing stats if this was the active codebase
            this.indexingStats = null;

            // Save snapshot after clearing index
            this.snapshotManager.saveCodebaseSnapshot();

            let resultText = `Successfully cleared codebase '${absolutePath}'`;

            const remainingIndexed = this.snapshotManager.getIndexedCodebases().length;
            const remainingIndexing = this.snapshotManager.getIndexingCodebases().length;

            if (remainingIndexed > 0 || remainingIndexing > 0) {
                resultText += `\n${remainingIndexed} other indexed codebase(s) and ${remainingIndexing} indexing codebase(s) remain`;
            }

            return {
                content: [{
                    type: "text",
                    text: resultText
                }]
            };
        } catch (error) {
            // Check if this is the collection limit error
            // Handle both direct string throws and Error objects containing the message
            const errorMessage = typeof error === 'string' ? error : (error instanceof Error ? error.message : String(error));

            if (errorMessage === COLLECTION_LIMIT_MESSAGE || errorMessage.includes(COLLECTION_LIMIT_MESSAGE)) {
                // Return the collection limit message as a successful response
                // This ensures LLM treats it as final answer, not as retryable error
                return {
                    content: [{
                        type: "text",
                        text: COLLECTION_LIMIT_MESSAGE
                    }]
                };
            }

            return {
                content: [{
                    type: "text",
                    text: `Error clearing index: ${errorMessage}`
                }],
                isError: true
            };
        }
    }

    public async handleGetIndexingStatus(args: any) {
        const { path: codebasePath } = args;

        try {
            // Force absolute path resolution
            const absolutePath = ensureAbsolutePath(codebasePath);

            // Validate path exists
            if (!fs.existsSync(absolutePath)) {
                return {
                    content: [{
                        type: "text",
                        text: `Error: Path '${absolutePath}' does not exist. Original input: '${codebasePath}'`
                    }],
                    isError: true
                };
            }

            // Check if it's a directory
            const stat = fs.statSync(absolutePath);
            if (!stat.isDirectory()) {
                return {
                    content: [{
                        type: "text",
                        text: `Error: Path '${absolutePath}' is not a directory`
                    }],
                    isError: true
                };
            }

            await this.syncIndexedCodebasesFromCloud();

            // Check indexing status using new status system
            const statusCodebasePath = this.snapshotManager.findTrackedCodebasePath(absolutePath) || absolutePath;
            const status = this.snapshotManager.getCodebaseStatus(statusCodebasePath);
            const info = this.snapshotManager.getCodebaseInfo(statusCodebasePath);

            let statusMessage = '';

            switch (status) {
                case 'indexed':
                    if (info && 'indexedFiles' in info) {
                        const indexedInfo = info as any;
                        statusMessage = `✅ Codebase '${statusCodebasePath}' is fully indexed and ready for search.`;
                        statusMessage += `\n📊 Statistics: ${indexedInfo.indexedFiles} files, ${indexedInfo.totalChunks} chunks`;
                        statusMessage += `\n📅 Status: ${indexedInfo.indexStatus}`;
                        statusMessage += `\n🕐 Last updated: ${new Date(indexedInfo.lastUpdated).toLocaleString()}`;
                    } else {
                        statusMessage = `✅ Codebase '${statusCodebasePath}' is fully indexed and ready for search.`;
                    }
                    break;

                case 'indexing':
                    if (info && 'indexingPercentage' in info) {
                        const indexingInfo = info as any;
                        const progressPercentage = indexingInfo.indexingPercentage || 0;
                        statusMessage = `🔄 Codebase '${statusCodebasePath}' is currently being indexed. Progress: ${progressPercentage.toFixed(1)}%`;

                        // Add more detailed status based on progress
                        if (progressPercentage < 10) {
                            statusMessage += ' (Preparing and scanning files...)';
                        } else if (progressPercentage < 100) {
                            statusMessage += ' (Processing files and generating embeddings...)';
                        }
                        statusMessage += `\n🕐 Last updated: ${new Date(indexingInfo.lastUpdated).toLocaleString()}`;
                    } else {
                        statusMessage = `🔄 Codebase '${statusCodebasePath}' is currently being indexed.`;
                    }
                    break;

                case 'indexfailed':
                    if (info && 'errorMessage' in info) {
                        const failedInfo = info as any;
                        statusMessage = `❌ Codebase '${statusCodebasePath}' indexing failed.`;
                        statusMessage += `\n🚨 Error: ${failedInfo.errorMessage}`;
                        if (failedInfo.lastAttemptedPercentage !== undefined) {
                            statusMessage += `\n📊 Failed at: ${failedInfo.lastAttemptedPercentage.toFixed(1)}% progress`;
                        }
                        statusMessage += `\n🕐 Failed at: ${new Date(failedInfo.lastUpdated).toLocaleString()}`;
                        statusMessage += `\n💡 You can retry indexing by running the index_codebase command again.`;
                    } else {
                        statusMessage = `❌ Codebase '${statusCodebasePath}' indexing failed. You can retry indexing.`;
                    }
                    break;

                case 'not_found':
                default:
                    statusMessage = `❌ Codebase '${absolutePath}' is not indexed. Please use the index_codebase tool to index it first.`;
                    break;
            }

            const pathInfo = codebasePath !== absolutePath
                ? `\nNote: Input path '${codebasePath}' was resolved to absolute path '${absolutePath}'`
                : '';
            const matchedPathInfo = statusCodebasePath !== absolutePath
                ? `\nRequested path '${absolutePath}' is covered by tracked codebase '${statusCodebasePath}'.`
                : '';

            return {
                content: [{
                    type: "text",
                    text: statusMessage + pathInfo + matchedPathInfo
                }]
            };

        } catch (error: any) {
            return {
                content: [{
                    type: "text",
                    text: `Error getting indexing status: ${error.message || error}`
                }],
                isError: true
            };
        }
    }

    /**
     * list_categories — enumerate the knowledge-base categories and how much of
     * each is actually searchable. A "category" is an immediate child directory
     * of the knowledge root (e.g. `haxe`, `godot`, `game_engine_books`); all
     * categories share the same Milvus collection(s) keyed by the root path,
     * distinguished by their `relativePath` prefix. We read the root's child
     * dirs (the canonical category set, which also surfaces dirs present on disk
     * but not yet indexed) and count indexed chunks per category straight from
     * Milvus — independent of the MCP snapshot registry, which is empty when the
     * corpus was indexed via the standalone infra/*-knowledge.js scripts.
     */
    public async handleListCategories(args: any) {
        const { path: rootPath } = args || {};
        try {
            const absRoot = ensureAbsolutePath(rootPath);

            if (!fs.existsSync(absRoot) || !fs.statSync(absRoot).isDirectory()) {
                return {
                    content: [{ type: "text", text: `Error: Path '${absRoot}' does not exist or is not a directory.` }],
                    isError: true
                };
            }

            const addr = this.context.getCollectionAddress(absRoot);
            const vdb = this.context.getVectorDatabase();

            // Collections to count against, deduped by name (in non-split mode
            // prose === code === legacy → a single "chunks" column).
            const wanted = addr.isSplit
                ? [{ label: 'code', name: addr.code }, { label: 'doc', name: addr.prose }]
                : [{ label: 'chunks', name: addr.legacy }];
            const cols: { label: string; name: string }[] = [];
            for (const c of wanted) {
                if (cols.some(x => x.name === c.name)) continue;
                if (await vdb.hasCollection(c.name)) cols.push(c);
            }

            if (cols.length === 0) {
                return {
                    content: [{
                        type: "text",
                        text: `No index exists for '${absRoot}' yet. Index it with the index_codebase tool or the standalone infra/{index,update}-knowledge.js scripts, then ask again.`
                    }]
                };
            }

            // Categories = immediate, non-dot child directories of the root.
            const categories = fs.readdirSync(absRoot, { withFileTypes: true })
                .filter(e => e.isDirectory() && !e.name.startsWith('.'))
                .map(e => e.name)
                .sort((a, b) => a.localeCompare(b));

            const countFor = async (collName: string, cat: string): Promise<number> => {
                try {
                    // `relativePath` values look like `<category>/...`; the `/`
                    // anchors the prefix so `go/%` can't match `golang/...`.
                    const rows = await vdb.query(collName, `relativePath like "${cat}/%"`, ["count(*)"]);
                    const raw = rows && rows[0] ? (rows[0]["count(*)"] ?? rows[0].count) : 0;
                    const n = typeof raw === 'string' ? parseInt(raw, 10) : Number(raw);
                    return Number.isFinite(n) ? n : 0;
                } catch {
                    return 0;
                }
            };

            const rows: { cat: string; counts: Record<string, number>; total: number }[] = [];
            for (const cat of categories) {
                const counts: Record<string, number> = {};
                let total = 0;
                for (const c of cols) {
                    const n = await countFor(c.name, cat);
                    counts[c.label] = n;
                    total += n;
                }
                rows.push({ cat, counts, total });
            }

            const indexed = rows.filter(r => r.total > 0);
            const pending = rows.filter(r => r.total === 0);

            let text = `Knowledge base categories under ${absRoot}\n`;
            text += `(search a category by calling search_code with path="${absRoot}" and naming it in the query)\n`;

            if (indexed.length === 0) {
                text += `\nNothing is indexed yet — every category dir on disk has 0 chunks.`;
            } else {
                text += `\nSearchable — ask about any of these:\n`;
                for (const r of indexed) {
                    const breakdown = cols.map(c => `${(r.counts[c.label] || 0).toLocaleString()} ${c.label}`).join(' + ');
                    text += `  • ${r.cat} — ${breakdown} chunks\n`;
                }
            }

            if (pending.length > 0) {
                text += `\nOn disk but NOT indexed (can't be searched until indexed):\n`;
                for (const r of pending) text += `  • ${r.cat}\n`;
            }

            text += `\n${indexed.length} searchable / ${rows.length} total categories.`;

            return { content: [{ type: "text", text }] };
        } catch (error: any) {
            return {
                content: [{ type: "text", text: `Error listing categories: ${error?.message || String(error)}` }],
                isError: true
            };
        }
    }
}
