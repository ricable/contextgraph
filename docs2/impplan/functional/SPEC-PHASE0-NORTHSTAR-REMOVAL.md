# Functional Specification: Phase 0 - North Star System Removal

```xml
<functional_spec id="SPEC-PHASE0" version="1.0">
<metadata>
  <title>North Star System Removal</title>
  <status>approved</status>
  <owner>Context Graph Team</owner>
  <created>2026-01-16</created>
  <last_updated>2026-01-16</last_updated>
  <implements>impplan.md Part 0</implements>
  <related_specs>
    <spec_ref>SPEC-PHASE1</spec_ref>
  </related_specs>
</metadata>

<overview>
Complete removal of the North Star alignment system and all associated components. The manual goal-setting paradigm (North Star) is being replaced with an emergent topic discovery system. This phase MUST complete before any other phases can begin, as the North Star components create incompatible 1024D vectors that conflict with the new 13-space teleological architecture.

**Problem Solved**: The North Star system requires manual goal definition (`set_north_star()`, `define_goal()`) which creates single-purpose 1024D vectors incompatible with the 13-embedding teleological fingerprints. The system also maintains `SELF_EGO_NODE` and Identity Continuity tracking that will be replaced by Topic Stability tracking.

**Who Benefits**: Developers working on the new emergent topic system; the system itself which can now operate autonomously without manual goal setting.
</overview>

<user_stories>
<story id="US-P0-01" priority="must-have">
  <narrative>
    As a system maintainer
    I want all North Star MCP tools removed
    So that the API surface is clean and doesn't expose deprecated functionality
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P0-01-01">
      <given>The system is running</given>
      <when>A client calls auto_bootstrap_north_star, get_alignment_drift, get_drift_history, trigger_drift_correction, get_identity_continuity, or get_ego_state</when>
      <then>The system returns a "tool not found" error</then>
    </criterion>
    <criterion id="AC-P0-01-02">
      <given>The MCP tool registry</given>
      <when>Listing all available tools</when>
      <then>None of the 6 removed tools appear in the list</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-P0-02" priority="must-have">
  <narrative>
    As a system architect
    I want all North Star data structures removed from the codebase
    So that there are no orphaned types or dead code paths
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P0-02-01">
      <given>The codebase</given>
      <when>Searching for NorthStar, GoalHierarchy, SelfEgoNode, IdentityContinuityMonitor, DriftDetector, DriftCorrector</when>
      <then>No struct definitions are found</then>
    </criterion>
    <criterion id="AC-P0-02-02">
      <given>The codebase</given>
      <when>Searching for north_star_alignment field</when>
      <then>No field definitions or usages are found</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-P0-03" priority="must-have">
  <narrative>
    As a developer
    I want the system to compile and pass all remaining tests
    So that I can verify the removal was clean
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P0-03-01">
      <given>The modified codebase</given>
      <when>Running cargo build --all</when>
      <then>Compilation succeeds with no errors</then>
    </criterion>
    <criterion id="AC-P0-03-02">
      <given>The modified codebase</given>
      <when>Running cargo test --all (excluding removed test files)</when>
      <then>All tests pass</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-P0-04" priority="must-have">
  <narrative>
    As a system operator
    I want the storage layer to NOT have ego_node column family
    So that the database schema matches the new architecture
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P0-04-01">
      <given>A fresh database initialization</given>
      <when>Listing RocksDB column families</when>
      <then>No "ego_node" column family exists</then>
    </criterion>
  </acceptance_criteria>
</story>
</user_stories>

<requirements>
<requirement id="REQ-P0-01" story_ref="US-P0-01" priority="must">
  <description>Delete MCP tool handlers: auto_bootstrap_north_star, get_alignment_drift, get_drift_history, trigger_drift_correction, get_identity_continuity, get_ego_state</description>
  <rationale>These tools expose the deprecated North Star alignment system</rationale>
  <files>
    <file action="DELETE">crates/context-graph-mcp/src/handlers/autonomous/bootstrap.rs</file>
    <file action="DELETE">crates/context-graph-mcp/src/handlers/autonomous/drift.rs</file>
    <file action="MODIFY">crates/context-graph-mcp/src/handlers/tools/dispatch.rs</file>
    <file action="MODIFY">crates/context-graph-mcp/src/handlers/tools/gwt_consciousness.rs</file>
    <file action="MODIFY">crates/context-graph-mcp/src/tools/names.rs</file>
    <file action="MODIFY">crates/context-graph-mcp/src/tools/definitions/autonomous.rs</file>
    <file action="MODIFY">crates/context-graph-mcp/src/tools/definitions/gwt.rs</file>
    <file action="MODIFY">crates/context-graph-mcp/src/tools/registry.rs</file>
  </files>
</requirement>

<requirement id="REQ-P0-02" story_ref="US-P0-02" priority="must">
  <description>Delete ego_node directory and all Identity Continuity components</description>
  <rationale>SelfEgoNode and IC monitoring are replaced by Topic Stability tracking</rationale>
  <files>
    <file action="DELETE">crates/context-graph-core/src/gwt/ego_node/ (entire directory)</file>
    <file action="MODIFY">crates/context-graph-core/src/gwt/mod.rs</file>
  </files>
</requirement>

<requirement id="REQ-P0-03" story_ref="US-P0-02" priority="must">
  <description>Delete drift detection and correction components</description>
  <rationale>Drift detection is replaced by divergence detection in the new system</rationale>
  <files>
    <file action="DELETE">crates/context-graph-core/src/autonomous/drift/ (entire directory)</file>
    <file action="DELETE">crates/context-graph-core/src/autonomous/services/drift_detector/ (entire directory)</file>
    <file action="DELETE">crates/context-graph-core/src/autonomous/services/drift_corrector/ (entire directory)</file>
    <file action="DELETE">crates/context-graph-core/src/autonomous/bootstrap.rs</file>
    <file action="MODIFY">crates/context-graph-core/src/autonomous/mod.rs</file>
  </files>
</requirement>

<requirement id="REQ-P0-04" story_ref="US-P0-04" priority="must">
  <description>Remove ego_node storage operations and column family</description>
  <rationale>Storage layer must not reference deleted types</rationale>
  <files>
    <file action="DELETE">crates/context-graph-storage/src/teleological/rocksdb_store/ego_node.rs</file>
    <file action="MODIFY">crates/context-graph-storage/src/teleological/serialization.rs</file>
    <file action="MODIFY">crates/context-graph-storage/src/teleological/column_families.rs</file>
    <file action="MODIFY">crates/context-graph-storage/src/teleological/rocksdb_store/trait_impl.rs</file>
    <file action="MODIFY">crates/context-graph-storage/src/teleological/mod.rs</file>
  </files>
</requirement>

<requirement id="REQ-P0-05" story_ref="US-P0-02" priority="must">
  <description>Remove theta_to_north_star field from TeleologicalFingerprint</description>
  <rationale>This field computes alignment to a North Star that no longer exists</rationale>
  <files>
    <file action="MODIFY">crates/context-graph-embeddings/src/storage/types/fingerprint.rs</file>
  </files>
</requirement>

<requirement id="REQ-P0-06" story_ref="US-P0-03" priority="must">
  <description>Delete or modify all test files that reference North Star components</description>
  <rationale>Tests must not reference deleted types</rationale>
  <files>
    <file action="DELETE">crates/context-graph-mcp/src/handlers/tests/north_star.rs</file>
    <file action="DELETE">crates/context-graph-mcp/src/handlers/tests/purpose/north_star_alignment.rs</file>
    <file action="DELETE">crates/context-graph-mcp/src/handlers/tests/purpose/north_star_update.rs</file>
    <file action="MODIFY">crates/context-graph-storage/tests/full_state_verification/persistence_tests.rs</file>
    <file action="MODIFY">crates/context-graph-storage/tests/full_state_verification/helpers.rs</file>
    <file action="MODIFY">crates/context-graph-storage/tests/teleological_integration.rs</file>
    <file action="MODIFY">crates/context-graph-mcp/src/handlers/tests/exhaustive_mcp_tools/identity_continuity.rs</file>
    <file action="MODIFY">crates/context-graph-mcp/src/handlers/tests/exhaustive_mcp_tools/autonomous_tools.rs</file>
    <file action="DELETE">crates/context-graph-storage/src/teleological/tests/ego_node.rs</file>
  </files>
</requirement>

<requirement id="REQ-P0-07" story_ref="US-P0-01" priority="must">
  <description>Update MCP handler core to remove North Star dependencies</description>
  <rationale>Handler infrastructure must not import deleted types</rationale>
  <files>
    <file action="MODIFY">crates/context-graph-mcp/src/handlers/core/handlers.rs</file>
    <file action="MODIFY">crates/context-graph-mcp/src/handlers/autonomous/mod.rs</file>
    <file action="MODIFY">crates/context-graph-mcp/src/handlers/autonomous/status.rs</file>
    <file action="MODIFY">crates/context-graph-mcp/src/handlers/autonomous/params.rs</file>
    <file action="MODIFY">crates/context-graph-mcp/src/handlers/purpose/drift.rs</file>
  </files>
</requirement>

<requirement id="REQ-P0-08" story_ref="US-P0-03" priority="must">
  <description>Update tool count in registry from 59 to 53</description>
  <rationale>Registry must accurately reflect available tools</rationale>
  <files>
    <file action="MODIFY">crates/context-graph-mcp/src/tools/registry.rs</file>
  </files>
</requirement>
</requirements>

<edge_cases>
<edge_case id="EC-P0-01" req_ref="REQ-P0-01">
  <scenario>External client caches old tool list and calls removed tool</scenario>
  <expected_behavior>System returns clear error: "Tool 'auto_bootstrap_north_star' not found. This tool has been deprecated and removed." - NOT a silent failure or panic</expected_behavior>
</edge_case>

<edge_case id="EC-P0-02" req_ref="REQ-P0-04">
  <scenario>Existing database has ego_node column family with data</scenario>
  <expected_behavior>System logs warning about orphaned data but continues operation. Migration path documented for data recovery if needed. System MUST NOT fail to start.</expected_behavior>
</edge_case>

<edge_case id="EC-P0-03" req_ref="REQ-P0-05">
  <scenario>Code references theta_to_north_star in serialization format</scenario>
  <expected_behavior>Serialization format version bumped. Old fingerprints can still be read (field ignored) but new fingerprints do not include field.</expected_behavior>
</edge_case>

<edge_case id="EC-P0-04" req_ref="REQ-P0-06">
  <scenario>Test helper creates mock with north_star_alignment field</scenario>
  <expected_behavior>Helper updated to remove field. If field required for backwards compat test, test deleted entirely.</expected_behavior>
</edge_case>
</edge_cases>

<error_states>
<error id="ERR-P0-01" http_code="404">
  <condition>Client calls removed MCP tool</condition>
  <message>Tool '[tool_name]' not found. This tool has been deprecated and removed in version X.X.</message>
  <recovery>Client should update to use new Topic Portfolio tools (when implemented in Phase 4)</recovery>
</error>

<error id="ERR-P0-02" http_code="500">
  <condition>Code references deleted type during compilation</condition>
  <message>Compilation error: cannot find type 'SelfEgoNode' in this scope</message>
  <recovery>Developer must update import paths and remove usages</recovery>
</error>
</error_states>

<test_plan>
<test_case id="TC-P0-01" type="integration" req_ref="REQ-P0-01">
  <description>Verify all 6 MCP tools return 404 when called</description>
  <inputs>["auto_bootstrap_north_star", "get_alignment_drift", "get_drift_history", "trigger_drift_correction", "get_identity_continuity", "get_ego_state"]</inputs>
  <expected>All return 404 "tool not found"</expected>
</test_case>

<test_case id="TC-P0-02" type="unit" req_ref="REQ-P0-02">
  <description>Verify no North Star types exist in codebase</description>
  <inputs>["grep -r 'struct NorthStar' crates/", "grep -r 'struct SelfEgoNode' crates/", "grep -r 'struct GoalHierarchy' crates/"]</inputs>
  <expected>All grep commands return no matches</expected>
</test_case>

<test_case id="TC-P0-03" type="integration" req_ref="REQ-P0-03">
  <description>Verify cargo build succeeds</description>
  <inputs>["cargo build --all"]</inputs>
  <expected>Exit code 0, no errors</expected>
</test_case>

<test_case id="TC-P0-04" type="integration" req_ref="REQ-P0-03">
  <description>Verify cargo test succeeds</description>
  <inputs>["cargo test --all"]</inputs>
  <expected>All tests pass (after removing North Star tests)</expected>
</test_case>

<test_case id="TC-P0-05" type="unit" req_ref="REQ-P0-04">
  <description>Verify ego_node column family not created</description>
  <inputs>["Initialize fresh RocksDB", "List column families"]</inputs>
  <expected>CF_EGO_NODE not in list</expected>
</test_case>

<test_case id="TC-P0-06" type="unit" req_ref="REQ-P0-05">
  <description>Verify TeleologicalFingerprint has no theta_to_north_star field</description>
  <inputs>["grep -r 'theta_to_north_star' crates/"]</inputs>
  <expected>No matches found</expected>
</test_case>

<test_case id="TC-P0-07" type="integration" req_ref="REQ-P0-08">
  <description>Verify MCP tool count is 53</description>
  <inputs>["Call MCP list_tools endpoint"]</inputs>
  <expected>Exactly 53 tools returned (59 - 6 = 53)</expected>
</test_case>
</test_plan>

<validation_criteria>
  <criterion>All 6 North Star MCP tools removed and return 404</criterion>
  <criterion>All North Star data structures deleted from codebase</criterion>
  <criterion>No compilation errors after removal</criterion>
  <criterion>All remaining tests pass</criterion>
  <criterion>ego_node column family not created on fresh DB</criterion>
  <criterion>theta_to_north_star field removed from fingerprint</criterion>
  <criterion>MCP tool count updated from 59 to 53</criterion>
</validation_criteria>
</functional_spec>
```

## Components Removed Summary

| Category | Component | Replacement |
|----------|-----------|-------------|
| Goal System | NorthStar struct | Emergent Topic Portfolio |
| Identity | SELF_EGO_NODE | Topic Profile (13D) |
| Identity | IdentityContinuityMonitor | Topic Stability Tracker |
| Drift | DriftDetector, DriftCorrector | Divergence Detection |
| Trigger | IC < 0.5 triggers dream | entropy > 0.7 + churn > 0.5 |

## Files Deleted (Complete List)

### Directories (DELETE RECURSIVELY)
1. `crates/context-graph-core/src/gwt/ego_node/` (~1929 lines)
2. `crates/context-graph-core/src/autonomous/drift/` (~1761 lines)
3. `crates/context-graph-core/src/autonomous/services/drift_detector/` (~850 lines)
4. `crates/context-graph-core/src/autonomous/services/drift_corrector/` (~500 lines)

### Individual Files
5. `crates/context-graph-core/src/autonomous/bootstrap.rs` (~6820 lines)
6. `crates/context-graph-mcp/src/handlers/autonomous/bootstrap.rs` (~412 lines)
7. `crates/context-graph-mcp/src/handlers/autonomous/drift.rs` (~500 lines)
8. `crates/context-graph-storage/src/teleological/rocksdb_store/ego_node.rs` (~300 lines)
9. `crates/context-graph-mcp/src/handlers/tests/north_star.rs` (~290 lines)
10. `crates/context-graph-mcp/src/handlers/tests/purpose/north_star_alignment.rs` (~200 lines)
11. `crates/context-graph-mcp/src/handlers/tests/purpose/north_star_update.rs` (~200 lines)
12. `crates/context-graph-storage/src/teleological/tests/ego_node.rs` (~150 lines)

**Total Lines Removed: ~13,012 lines**
