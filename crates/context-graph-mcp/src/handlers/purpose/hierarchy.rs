//! Goal hierarchy query handler.
//!
//! Handles the `goal/hierarchy_query` MCP method for navigating
//! and querying the goal hierarchy.

use serde_json::json;
use tracing::{error, instrument};
use uuid::Uuid;

use crate::handlers::Handlers;
use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::helpers::{compute_hierarchy_stats, goal_to_json};

impl Handlers {
    /// Handle goal/hierarchy_query request.
    ///
    /// Navigate and query the goal hierarchy.
    ///
    /// # Request Parameters
    /// - `operation` (required): "get_children", "get_ancestors", "get_subtree", "get_all", "get_goal"
    /// - `goal_id` (optional): Goal ID for targeted operations
    /// - `level` (optional): Filter by GoalLevel ("NorthStar", "Strategic", "Tactical", "Immediate")
    ///
    /// # Response
    /// - `goals`: Array of goal objects with hierarchy info
    /// - `hierarchy_stats`: Statistics about the hierarchy
    ///
    /// # Error Codes
    /// - INVALID_PARAMS (-32602): Invalid operation or goal_id
    /// - GOAL_NOT_FOUND (-32020): Goal ID not found
    #[instrument(skip(self, params), fields(method = "goal/hierarchy_query"))]
    pub(in crate::handlers) async fn handle_goal_hierarchy_query(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                error!("goal/hierarchy_query: Missing parameters");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters - operation required",
                );
            }
        };

        let operation = match params.get("operation").and_then(|v| v.as_str()) {
            Some(op) => op,
            None => {
                error!("goal/hierarchy_query: Missing 'operation' parameter");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing required 'operation' parameter. Valid: get_children, get_ancestors, get_subtree, get_all, get_goal",
                );
            }
        };

        let hierarchy = self.goal_hierarchy.read();

        match operation {
            "get_all" => {
                let goals: Vec<serde_json::Value> =
                    hierarchy.iter().map(goal_to_json).collect();

                let stats = compute_hierarchy_stats(&hierarchy);

                JsonRpcResponse::success(
                    id,
                    json!({
                        "goals": goals,
                        "count": goals.len(),
                        "hierarchy_stats": stats
                    }),
                )
            }

            "get_goal" => {
                let goal_id = match params.get("goal_id").and_then(|v| v.as_str()) {
                    Some(gid) => match Uuid::parse_str(gid) {
                        Ok(uuid) => uuid,
                        Err(_) => {
                            error!("goal/hierarchy_query: Invalid goal_id UUID format: {}", gid);
                            return JsonRpcResponse::error(
                                id,
                                error_codes::INVALID_PARAMS,
                                format!("Invalid goal_id UUID format: {}", gid),
                            );
                        }
                    },
                    None => {
                        error!("goal/hierarchy_query: get_goal requires 'goal_id'");
                        return JsonRpcResponse::error(
                            id,
                            error_codes::INVALID_PARAMS,
                            "get_goal operation requires 'goal_id' parameter",
                        );
                    }
                };

                match hierarchy.get(&goal_id) {
                    Some(goal) => {
                        JsonRpcResponse::success(id, json!({ "goal": goal_to_json(goal) }))
                    }
                    None => {
                        error!(goal_id = %goal_id, "goal/hierarchy_query: Goal not found");
                        JsonRpcResponse::error(
                            id,
                            error_codes::GOAL_NOT_FOUND,
                            format!("Goal not found: {}", goal_id),
                        )
                    }
                }
            }

            "get_children" => {
                let goal_id = match params.get("goal_id").and_then(|v| v.as_str()) {
                    Some(gid) => match Uuid::parse_str(gid) {
                        Ok(uuid) => uuid,
                        Err(_) => {
                            error!("goal/hierarchy_query: Invalid goal_id UUID format: {}", gid);
                            return JsonRpcResponse::error(
                                id,
                                error_codes::INVALID_PARAMS,
                                format!("Invalid goal_id UUID format: {}", gid),
                            );
                        }
                    },
                    None => {
                        error!("goal/hierarchy_query: get_children requires 'goal_id'");
                        return JsonRpcResponse::error(
                            id,
                            error_codes::INVALID_PARAMS,
                            "get_children operation requires 'goal_id' parameter",
                        );
                    }
                };

                // Verify parent exists
                if hierarchy.get(&goal_id).is_none() {
                    error!(goal_id = %goal_id, "goal/hierarchy_query: Parent goal not found");
                    return JsonRpcResponse::error(
                        id,
                        error_codes::GOAL_NOT_FOUND,
                        format!("Parent goal not found: {}", goal_id),
                    );
                }

                let children: Vec<serde_json::Value> = hierarchy
                    .children(&goal_id)
                    .into_iter()
                    .map(goal_to_json)
                    .collect();

                JsonRpcResponse::success(
                    id,
                    json!({
                        "parent_goal_id": goal_id.to_string(),
                        "children": children,
                        "count": children.len()
                    }),
                )
            }

            "get_ancestors" => {
                let goal_id = match params.get("goal_id").and_then(|v| v.as_str()) {
                    Some(gid) => match Uuid::parse_str(gid) {
                        Ok(uuid) => uuid,
                        Err(_) => {
                            error!("goal/hierarchy_query: Invalid goal_id UUID format: {}", gid);
                            return JsonRpcResponse::error(
                                id,
                                error_codes::INVALID_PARAMS,
                                format!("Invalid goal_id UUID format: {}", gid),
                            );
                        }
                    },
                    None => {
                        error!("goal/hierarchy_query: get_ancestors requires 'goal_id'");
                        return JsonRpcResponse::error(
                            id,
                            error_codes::INVALID_PARAMS,
                            "get_ancestors operation requires 'goal_id' parameter",
                        );
                    }
                };

                let path = hierarchy.path_to_north_star(&goal_id);
                if path.is_empty() {
                    error!(goal_id = %goal_id, "goal/hierarchy_query: Goal not found for ancestors");
                    return JsonRpcResponse::error(
                        id,
                        error_codes::GOAL_NOT_FOUND,
                        format!("Goal not found: {}", goal_id),
                    );
                }

                let ancestors: Vec<serde_json::Value> = path
                    .iter()
                    .filter_map(|gid| hierarchy.get(gid))
                    .map(goal_to_json)
                    .collect();

                JsonRpcResponse::success(
                    id,
                    json!({
                        "goal_id": goal_id.to_string(),
                        "ancestors": ancestors,
                        "depth": ancestors.len()
                    }),
                )
            }

            "get_subtree" => {
                let goal_id = match params.get("goal_id").and_then(|v| v.as_str()) {
                    Some(gid) => match Uuid::parse_str(gid) {
                        Ok(uuid) => uuid,
                        Err(_) => {
                            error!("goal/hierarchy_query: Invalid goal_id UUID format: {}", gid);
                            return JsonRpcResponse::error(
                                id,
                                error_codes::INVALID_PARAMS,
                                format!("Invalid goal_id UUID format: {}", gid),
                            );
                        }
                    },
                    None => {
                        error!("goal/hierarchy_query: get_subtree requires 'goal_id'");
                        return JsonRpcResponse::error(
                            id,
                            error_codes::INVALID_PARAMS,
                            "get_subtree operation requires 'goal_id' parameter",
                        );
                    }
                };

                // Get root of subtree
                let root = match hierarchy.get(&goal_id) {
                    Some(g) => g,
                    None => {
                        error!(goal_id = %goal_id, "goal/hierarchy_query: Subtree root not found");
                        return JsonRpcResponse::error(
                            id,
                            error_codes::GOAL_NOT_FOUND,
                            format!("Subtree root not found: {}", goal_id),
                        );
                    }
                };

                // Collect subtree using BFS
                let mut subtree = vec![goal_to_json(root)];
                let mut queue = vec![goal_id];

                while let Some(current_id) = queue.pop() {
                    for child in hierarchy.children(&current_id) {
                        subtree.push(goal_to_json(child));
                        queue.push(child.id);
                    }
                }

                JsonRpcResponse::success(
                    id,
                    json!({
                        "root_goal_id": goal_id.to_string(),
                        "subtree": subtree,
                        "count": subtree.len()
                    }),
                )
            }

            _ => {
                error!(
                    operation = operation,
                    "goal/hierarchy_query: Unknown operation"
                );
                JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!(
                        "Unknown operation '{}'. Valid: get_children, get_ancestors, get_subtree, get_all, get_goal",
                        operation
                    ),
                )
            }
        }
    }
}
