# TASK-P0-005: Verify Removal and Run Tests

```xml
<task_spec id="TASK-P0-005" version="1.0">
<metadata>
  <title>Verify Complete Removal and Run Tests</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>5</sequence>
  <phase>0</phase>
  <implements>
    <requirement_ref>REQ-P0-06</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P0-004</task_ref>
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
</metadata>

<context>
Final task in Phase 0. Verifies that all North Star components have been completely
removed and that the remaining system compiles and passes tests.

This is the verification gate before proceeding to Phase 1. If any North Star
references remain or tests fail, the removal is incomplete.

CRITICAL: System must compile and core tests must pass. The system works or fails fast.
No partial functionality tolerated.
</context>

<input_context_files>
  <file purpose="removal_checklist">docs2/impplan/technical/TECH-PHASE0-NORTHSTAR-REMOVAL.md#verification</file>
  <file purpose="spec_requirements">docs2/impplan/functional/SPEC-PHASE0-NORTHSTAR-REMOVAL.md</file>
</input_context_files>

<prerequisites>
  <check>TASK-P0-001 through P0-004 completed</check>
  <check>All file deletions staged in git</check>
  <check>All modifications staged in git</check>
</prerequisites>

<scope>
  <in_scope>
    - Verify no North Star references remain in code
    - Verify no drift references remain in code
    - Verify no identity_continuity references remain
    - Verify no self_ego references remain
    - Fix any remaining compilation errors from removal
    - Run full test suite (excluding deleted tests)
    - Create verification report
    - Create atomic git commit for Phase 0
  </in_scope>
  <out_of_scope>
    - Adding new functionality
    - Phase 1 implementation
  </out_of_scope>
</scope>

<definition_of_done>
  <verification_checks>
    <check name="no_north_star_refs">grep -r "north_star" src/ returns empty</check>
    <check name="no_drift_refs">grep -r "DriftDetector\|DriftCorrector" src/ returns empty</check>
    <check name="no_identity_refs">grep -r "identity_continuity\|IdentityContinuity" src/ returns empty</check>
    <check name="no_ego_refs">grep -r "SelfEgoNode\|self_ego" src/ returns empty</check>
    <check name="compilation">cargo build succeeds</check>
    <check name="tests">cargo test passes (excluding North Star tests)</check>
  </verification_checks>

  <constraints>
    - ALL grep checks must return empty
    - Build must succeed with no errors
    - Core tests must pass
    - Create single atomic commit for all Phase 0 changes
  </constraints>

  <verification>
    - 7-step verification protocol from tech spec completed
    - Verification report generated
    - Git commit created with proper message
  </verification>
</definition_of_done>

<pseudo_code>
Verification sequence:
1. Run comprehensive grep checks:
   grep -r "north_star" src/ crates/
   grep -r "NorthStar" src/ crates/
   grep -r "DriftDetector" src/ crates/
   grep -r "DriftCorrector" src/ crates/
   grep -r "IdentityContinuity" src/ crates/
   grep -r "SelfEgoNode" src/ crates/
   grep -r "self_ego" src/ crates/

2. If any grep returns results:
   - Identify remaining references
   - Remove or update as needed
   - Re-run grep checks

3. Run compilation check:
   cargo build --all-targets

4. If compilation fails:
   - Identify error locations
   - Fix dangling imports/references
   - Re-run build

5. Run test suite:
   cargo test --all

6. If tests fail:
   - Identify failing tests
   - If test references North Star: delete test
   - If test is core functionality: fix issue
   - Re-run tests

7. Create verification report documenting:
   - All grep checks passed
   - Build succeeded
   - Test results

8. Create git commit:
   git add -A
   git commit -m "feat(phase0): complete North Star removal

   BREAKING CHANGE: North Star, SelfEgoNode, IdentityContinuity,
   DriftDetector, and DriftCorrector have been removed.

   - Deleted ~13,000 lines of North Star code
   - Removed MCP handlers for North Star tools
   - Updated constitution.yaml
   - Dropped North Star database tables
   - All verification checks passed

   Implements: TASK-P0-001 through TASK-P0-005"
</pseudo_code>

<files_to_modify>
  <file path="(any remaining references)">Fix dangling imports/references</file>
</files_to_modify>

<validation_criteria>
  <criterion>grep "north_star" returns no results in src/</criterion>
  <criterion>grep "NorthStar" returns no results in src/</criterion>
  <criterion>grep "DriftDetector" returns no results in src/</criterion>
  <criterion>grep "IdentityContinuity" returns no results in src/</criterion>
  <criterion>grep "SelfEgoNode" returns no results in src/</criterion>
  <criterion>cargo build succeeds with no errors</criterion>
  <criterion>cargo test passes core tests</criterion>
  <criterion>Git commit created for Phase 0</criterion>
</validation_criteria>

<test_commands>
  <command description="Check north_star gone">grep -r "north_star" src/ crates/ 2>/dev/null | grep -v target/ || echo "✓ No north_star references"</command>
  <command description="Check NorthStar gone">grep -r "NorthStar" src/ crates/ 2>/dev/null | grep -v target/ || echo "✓ No NorthStar references"</command>
  <command description="Check drift gone">grep -r "DriftDetector\|DriftCorrector" src/ crates/ 2>/dev/null | grep -v target/ || echo "✓ No Drift references"</command>
  <command description="Check identity gone">grep -r "IdentityContinuity" src/ crates/ 2>/dev/null | grep -v target/ || echo "✓ No IdentityContinuity references"</command>
  <command description="Check ego gone">grep -r "SelfEgoNode\|self_ego" src/ crates/ 2>/dev/null | grep -v target/ || echo "✓ No SelfEgo references"</command>
  <command description="Full build">cargo build --all-targets 2>&1 | tail -10</command>
  <command description="Run tests">cargo test --all 2>&1 | tail -20</command>
</test_commands>

<verification_report_template>
```markdown
# Phase 0 Verification Report

## Date: [YYYY-MM-DD]

## Grep Verification
| Pattern | Result |
|---------|--------|
| north_star | ✓ None found |
| NorthStar | ✓ None found |
| DriftDetector | ✓ None found |
| DriftCorrector | ✓ None found |
| IdentityContinuity | ✓ None found |
| SelfEgoNode | ✓ None found |
| self_ego | ✓ None found |

## Build Status
- cargo build: ✓ SUCCESS
- Warnings: [count]
- Errors: 0

## Test Results
- Total tests: [count]
- Passed: [count]
- Failed: 0
- Skipped: [count]

## Files Changed
- Deleted: [count] files
- Modified: [count] files
- Lines removed: ~13,000

## Git Commit
- Hash: [commit hash]
- Message: feat(phase0): complete North Star removal

## Conclusion
Phase 0 complete. System ready for Phase 1.
```
</verification_report_template>

<notes>
  <note category="false_positives">
    Grep may find "north_star" in documentation, comments, or git history.
    Focus on src/ and crates/ directories. Exclude target/ directory.
  </note>
  <note category="test_failures">
    Some tests may fail because they tested North Star functionality.
    These tests should have been deleted in P0-001. If found, delete them.
  </note>
  <note category="commit">
    Create a single atomic commit for all Phase 0 changes.
    This allows easy rollback if issues are discovered.
  </note>
</notes>
</task_spec>
```

## Execution Checklist

- [ ] Run all grep verification checks
- [ ] Fix any remaining references found
- [ ] Run cargo build --all-targets
- [ ] Fix any compilation errors
- [ ] Run cargo test --all
- [ ] Fix or delete any failing tests
- [ ] Generate verification report
- [ ] Stage all changes: git add -A
- [ ] Create atomic commit for Phase 0
- [ ] Mark Phase 0 complete
- [ ] Proceed to Phase 1
