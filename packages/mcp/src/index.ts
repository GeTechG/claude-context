#!/usr/bin/env node

// CRITICAL: Redirect console outputs to stderr IMMEDIATELY to avoid interfering with MCP JSON protocol
// Only MCP protocol messages should go to stdout
const originalConsoleLog = console.log;
const originalConsoleWarn = console.warn;

console.log = (...args: any[]) => {
    process.stderr.write('[LOG] ' + args.join(' ') + '\n');
};

console.warn = (...args: any[]) => {
    process.stderr.write('[WARN] ' + args.join(' ') + '\n');
};

// console.error already goes to stderr by default

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
    ListToolsRequestSchema,
    CallToolRequestSchema
} from "@modelcontextprotocol/sdk/types.js";
import { Context } from "@zilliz/claude-context-core";
import { MilvusVectorDatabase } from "@zilliz/claude-context-core";

// Import our modular components
import { createMcpConfig, logConfigurationSummary, showHelpMessage, ContextMcpConfig } from "./config.js";
import { createEmbeddingInstance, createProseEmbeddingInstance, logEmbeddingProviderInfo } from "./embedding.js";
import { createRerankerInstance } from "./reranker.js";
import { SnapshotManager } from "./snapshot.js";
import { SyncManager } from "./sync.js";
import { ToolHandlers } from "./handlers.js";

class ContextMcpServer {
    private server: Server;
    private context: Context;
    private snapshotManager: SnapshotManager;
    private syncManager: SyncManager;
    private toolHandlers: ToolHandlers;

    constructor(config: ContextMcpConfig) {
        // Initialize MCP server
        this.server = new Server(
            {
                name: config.name,
                version: config.version
            },
            {
                capabilities: {
                    tools: {}
                }
            }
        );

        // Initialize embedding provider
        console.log(`[EMBEDDING] Initializing embedding provider: ${config.embeddingProvider}`);
        console.log(`[EMBEDDING] Using model: ${config.embeddingModel}`);

        const embedding = createEmbeddingInstance(config);
        logEmbeddingProviderInfo(config, embedding);

        // prose-embedding-swap: optional distinct prose-pool dense embedder.
        // undefined → prose shares `embedding` (default bge-m3, byte-identical).
        const proseEmbedding = createProseEmbeddingInstance(config);

        // Initialize vector database
        const vectorDatabase = new MilvusVectorDatabase({
            address: config.milvusAddress,
            ...(config.milvusToken && { token: config.milvusToken })
        });

        // Initialize reranker (optional — Phase 2)
        const reranker = createRerankerInstance(config);

        // Initialize Claude Context
        this.context = new Context({
            embedding,
            ...(proseEmbedding && { proseEmbedding }),
            vectorDatabase,
            collectionNameOverride: config.collectionNameOverride,
            ...(reranker && { reranker })
        });

        // Initialize managers
        this.snapshotManager = new SnapshotManager();
        this.syncManager = new SyncManager(this.context, this.snapshotManager);
        this.toolHandlers = new ToolHandlers(this.context, this.snapshotManager);

        // Load existing codebase snapshot on startup
        this.snapshotManager.loadCodebaseSnapshot();

        this.setupTools();
    }

    private setupTools() {
        const index_description = `
Index a codebase directory to enable semantic search using a configurable code splitter.

⚠️ **IMPORTANT**:
- You MUST provide an absolute path to the target codebase.

✨ **Usage Guidance**:
- This tool is typically used when search fails due to an unindexed codebase.
- If indexing is attempted on an already indexed path, and a conflict is detected, you MUST prompt the user to confirm whether to proceed with a force index (i.e., re-indexing and overwriting the previous index).
`;


        const search_description = `
Search the indexed codebase using natural language queries within a specified absolute path.

⚠️ **IMPORTANT**:
- You MUST provide an absolute path.

🎯 **When to Use**:
This tool is versatile and can be used before completing various tasks to retrieve relevant context:
- **Code search**: Find specific functions, classes, or implementations
- **Context-aware assistance**: Gather relevant code context before making changes
- **Issue identification**: Locate problematic code sections or bugs
- **Code review**: Understand existing implementations and patterns
- **Refactoring**: Find all related code pieces that need to be updated
- **Feature development**: Understand existing architecture and similar implementations
- **Duplicate detection**: Identify redundant or duplicated code patterns across the codebase

✨ **Usage Guidance**:
- If the codebase is not indexed, this tool will return a clear error message indicating that indexing is required first.
- You can then use the index_codebase tool to index the codebase before searching again.
`;

        // Define available tools
        this.server.setRequestHandler(ListToolsRequestSchema, async () => {
            return {
                tools: [
                    {
                        name: "index_codebase",
                        description: index_description,
                        inputSchema: {
                            type: "object",
                            properties: {
                                path: {
                                    type: "string",
                                    description: `ABSOLUTE path to the codebase directory to index.`
                                },
                                force: {
                                    type: "boolean",
                                    description: "Force re-indexing even if already indexed",
                                    default: false
                                },
                                splitter: {
                                    type: "string",
                                    description: "Code splitter to use: 'ast' for syntax-aware splitting with automatic fallback, 'langchain' for character-based splitting",
                                    enum: ["ast", "langchain"],
                                    default: "ast"
                                },
                                customExtensions: {
                                    type: "array",
                                    items: {
                                        type: "string"
                                    },
                                    description: "Optional: Additional file extensions to include beyond defaults (e.g., ['.vue', '.svelte', '.astro']). Extensions should include the dot prefix or will be automatically added",
                                    default: []
                                },
                                ignorePatterns: {
                                    type: "array",
                                    items: {
                                        type: "string"
                                    },
                                    description: "Optional: Additional ignore patterns to exclude specific files/directories beyond defaults. Only include this parameter if the user explicitly requests custom ignore patterns (e.g., ['static/**', '*.tmp', 'private/**'])",
                                    default: []
                                }
                            },
                            required: ["path"]
                        }
                    },
                    {
                        name: "search_code",
                        description: search_description,
                        inputSchema: {
                            type: "object",
                            properties: {
                                path: {
                                    type: "string",
                                    description: `ABSOLUTE path to the codebase directory to search in.`
                                },
                                query: {
                                    type: "string",
                                    description: "Natural language query to search for in the codebase"
                                },
                                limit: {
                                    type: "number",
                                    description: "Maximum number of results to return",
                                    default: 10,
                                    maximum: 50
                                },
                                extensionFilter: {
                                    type: "array",
                                    items: {
                                        type: "string"
                                    },
                                    description: "Optional: List of file extensions to filter results. (e.g., ['.ts','.py']).",
                                    default: []
                                },
                                shapeHint: {
                                    type: "string",
                                    description: "Optional: query_shape hint forwarded to the retrieval router. Use 'concept' for broad reference/synthesis subqueries that need content-type diversity (activates the concept-span quota). Omit for ordinary identifier/descriptive lookups — behaviour then matches the pre-change one-shot path.",
                                    enum: ["single", "multi-hop", "comparison", "concept"]
                                }
                            },
                            required: ["path", "query"]
                        }
                    },
                    {
                        name: "clear_index",
                        description: `Clear the search index. IMPORTANT: You MUST provide an absolute path.`,
                        inputSchema: {
                            type: "object",
                            properties: {
                                path: {
                                    type: "string",
                                    description: `ABSOLUTE path to the codebase directory to clear.`
                                }
                            },
                            required: ["path"]
                        }
                    },
                    {
                        name: "get_indexing_status",
                        description: `Get the current indexing status of a codebase. Shows progress percentage for actively indexing codebases and completion status for indexed codebases.`,
                        inputSchema: {
                            type: "object",
                            properties: {
                                path: {
                                    type: "string",
                                    description: `ABSOLUTE path to the codebase directory to check status for.`
                                }
                            },
                            required: ["path"]
                        }
                    },
                    {
                        name: "expand_context",
                        description: `Follow the deterministic prose-graph to pull in a chunk's explicit 1-hop neighbours when a search_code answer is incomplete. Call it after search_code when you need the connected structure around a result — the parent/child sections, the code example next to a passage, co-mentioned chunks, the next chunk in the file, or an internally-linked page. Pass the \`Chunk-ID\` line from the search_code result as \`chunk_id\`. Returns each neighbour's relationship type (heading / code_example / co_mention / sequence / link), edge weight, location, heading path, content_type, and content. Optionally narrow with \`edge_types\` (e.g. ['heading','code_example'] for "this section and its examples") and bound with \`limit\`. This does NOT change retrieval ranking; it is a read-only navigation supplement to search_code. If the prose-graph side-index is unavailable it says so without affecting search.`,
                        inputSchema: {
                            type: "object",
                            properties: {
                                path: {
                                    type: "string",
                                    description: `ABSOLUTE path to the indexed codebase the chunk belongs to (same path used for search_code).`
                                },
                                chunk_id: {
                                    type: "string",
                                    description: "The Chunk-ID of the result to expand from, copied from a search_code result's `Chunk-ID:` line."
                                },
                                edge_types: {
                                    type: "array",
                                    items: {
                                        type: "string",
                                        enum: ["heading", "code_example", "co_mention", "sequence", "link"]
                                    },
                                    description: "Optional: restrict neighbours to these relationship types. Omit to return all types."
                                },
                                limit: {
                                    type: "integer",
                                    description: "Optional: maximum number of neighbours to return (default 10, max 50).",
                                    default: 10,
                                    maximum: 50
                                }
                            },
                            required: ["path", "chunk_id"]
                        }
                    },
                    {
                        name: "record_answer",
                        description: `Record the final answer you synthesized for a reference/knowledge-base task, so real usage can be analyzed and mined for future training. Call this ONCE at the END of a reference-answer task, after you have written your answer to the user. Pass the final answer text, the \`request_ids\` you drew on (copy the \`Request-ID:\` line from each search_code / expand_context result whose context you actually used), and — when you can judge it — an in-session \`signal\` (\`helpful\` if the retrieved context was sufficient and the answer is solid, \`not_helpful\` if retrieval missed or the answer is weak, \`unknown\` otherwise) plus a short \`note\`. This is observation only: it does not change retrieval or your answer. Safe to skip if no Request-ID was surfaced (usage logging off).`,
                        inputSchema: {
                            type: "object",
                            properties: {
                                answer: {
                                    type: "string",
                                    description: "The final synthesized answer text you delivered to the user."
                                },
                                request_ids: {
                                    type: "array",
                                    items: {
                                        type: "string"
                                    },
                                    description: "The Request-ID(s) of the search_code / expand_context calls whose retrieved context fed this answer (from each result's `Request-ID:` line)."
                                },
                                signal: {
                                    type: "string",
                                    description: "Optional in-session quality signal: 'helpful' if retrieval was sufficient and the answer is solid, 'not_helpful' if retrieval missed or the answer is weak, 'unknown' if unsure.",
                                    enum: ["helpful", "not_helpful", "unknown"]
                                },
                                note: {
                                    type: "string",
                                    description: "Optional short free-text note on what worked or what was missing (e.g. 'top result was off-topic, expand_context recovered it')."
                                },
                                query: {
                                    type: "string",
                                    description: "Optional: the user-facing question this answer addressed, when it differs from the raw search query."
                                }
                            },
                            required: ["answer", "request_ids"]
                        }
                    },
                ]
            };
        });

        // Handle tool execution
        this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
            const { name, arguments: args } = request.params;

            switch (name) {
                case "index_codebase":
                    return await this.toolHandlers.handleIndexCodebase(args);
                case "search_code":
                    return await this.toolHandlers.handleSearchCode(args);
                case "expand_context":
                    return await this.toolHandlers.handleExpandContext(args);
                case "record_answer":
                    return await this.toolHandlers.handleRecordAnswer(args);
                case "clear_index":
                    return await this.toolHandlers.handleClearIndex(args);
                case "get_indexing_status":
                    return await this.toolHandlers.handleGetIndexingStatus(args);

                default:
                    throw new Error(`Unknown tool: ${name}`);
            }
        });
    }

    async start() {
        console.log('[SYNC-DEBUG] MCP server start() method called');
        console.log('Starting Context MCP server...');

        // One-shot startup healing for legacy 0/0+completed snapshot entries
        // left over from pre-fix MCP versions. Runs before the transport accepts
        // requests so clients never observe the poisoning state. See Issue #295.
        await this.toolHandlers.validateLegacyZeroEntries();

        const transport = new StdioServerTransport();
        console.log('[SYNC-DEBUG] StdioServerTransport created, attempting server connection...');

        await this.server.connect(transport);
        console.log("MCP server started and listening on stdio.");
        console.log('[SYNC-DEBUG] Server connection established successfully');

        // Start background sync after server is connected
        console.log('[SYNC-DEBUG] Initializing background sync...');
        this.syncManager.startBackgroundSync();
        console.log('[SYNC-DEBUG] MCP server initialization complete');
    }
}

// Main execution
async function main() {
    // Parse command line arguments
    const args = process.argv.slice(2);

    // Show help if requested
    if (args.includes('--help') || args.includes('-h')) {
        showHelpMessage();
        process.exit(0);
    }

    // Create configuration
    const config = createMcpConfig();
    logConfigurationSummary(config);

    const server = new ContextMcpServer(config);
    await server.start();
}

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.error("Received SIGINT, shutting down gracefully...");
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.error("Received SIGTERM, shutting down gracefully...");
    process.exit(0);
});

// Always start the server - this is designed to be the main entry point
main().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
});
