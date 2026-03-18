# AI Data Availability Architecture and Implementation Guide

## Purpose

This document defines how AI should access accurate student and course data so responses stay correct for:

- Track, module, lesson visibility
- Progress and assessment state
- Opportunities and job-board personalization

It also defines what to mirror into AI DB versus what to fetch live from LMS.

## Problem Statement

Current behavior shows gaps where AI can return partial or stale answers because:

- Course content is available locally, but user progress and enrollment context are not always fetched live.
- Some data classes are cached too long for conversational correctness.
- Personalization for opportunities can drift when location and enrollment context are incomplete.

## Data Ownership Model

- LMS is source of truth for user state and progression.
- AI DB is source of truth for mirrored curriculum content retrieval.
- AI runtime composes both at response time.

## Endpoint Decision Matrix

### Mirror to AI DB

Use scheduled sync plus event-driven refresh where possible.

| Endpoint                                                                                             | Why Mirror                           |
| ---------------------------------------------------------------------------------------------------- | ------------------------------------ |
| /api/v1/courses                                                                                      | Core catalog, high read frequency    |
| /api/v1/courses/{id}                                                                                 | Canonical course metadata            |
| /api/v1/courses/c/{slug}                                                                             | Frontend slug-based route support    |
| /api/v1/courses/{courseId}/levels                                                                    | Hierarchical curriculum structure    |
| /api/v1/courses/{courseId}/modules                                                                   | Module-level retrieval               |
| /api/v1/courses/{courseId}/lessons                                                                   | Lesson-level retrieval               |
| /api/v1/courses/{courseId}/materials                                                                 | Resource retrieval for AI answers    |
| /api/v1/courses/{courseId}/levels/{levelTitle}/modules/{moduleIndex}/lessons/{lessonIndex}/materials | Deep lesson artifacts and references |

### Call Live via Tool Calls

Always fetch live for user-specific, volatile state.

| Endpoint                                                        | Why Live                                        |
| --------------------------------------------------------------- | ----------------------------------------------- |
| /api/v1/user/profile                                            | Current user identity and account state         |
| /api/v1/ai-mentor/onboarding/me                                 | Rich learner context used for personalization   |
| /api/v1/ai-mentor/onboarding/status                             | Wizard completion and flow gating               |
| /api/v1/subscription/me                                         | Active plan and track entitlements              |
| /api/v1/enrollment/active                                       | Active enrollments and track access             |
| /api/v1/enrollment/student/blackboard                           | Dashboard-grade enrollment/progress summary     |
| /api/v1/enrollment/{courseId}/structure                         | Locked/unlocked and lesson progression shape    |
| /api/v1/enrollment/{courseId}/progress                          | Full progress breakdown per level/module/lesson |
| /api/v1/enrollment/course/{courseId}/lesson/{lessonId}/progress | Real-time lesson progress state                 |
| /api/v1/enrollment/{studentId}/attempts                         | Assessment attempts and result context          |
| /api/v1/enrollment/module/{moduleTitle}/submit-assessment       | Assessment transaction context                  |
| /api/v1/enrollment/level/{levelTitle}/submit-assessment         | Assessment transaction context                  |

### Do Not Mirror as Primary Source

These are transactional endpoints. Keep LMS authoritative.

| Endpoint Group                          | Policy                           |
| --------------------------------------- | -------------------------------- |
| Assessment write endpoints              | Live only, no mirrored authority |
| Project submit/review endpoints         | Live only, no mirrored authority |
| Lesson progress update endpoints        | Live only, no mirrored authority |
| Subscription/payment mutating endpoints | Live only                        |

## Runtime Data Composition

### Content Plane

- Source: Mirrored AI DB content indices.
- Use: Retrieval for explanations, summaries, module and lesson content references.

### User State Plane

- Source: Live LMS tool calls per run.
- Use: Enrollment checks, progress claims, assessments, personalization.

### Response Rule

AI must combine content plane plus user state plane before making definitive statements about:

- What the student can access
- What the student has completed
- What to do next in track progression

## Required AI Tool Surface

Add or formalize these tools:

- get_student_enrollment_overview
  - Endpoint: /api/v1/enrollment/student/blackboard
  - Purpose: quick track and progress overview.

- get_course_structure
  - Endpoint: /api/v1/enrollment/{courseId}/structure
  - Purpose: module and lesson unlocked/completed map.

- get_course_progress
  - Endpoint: /api/v1/enrollment/{courseId}/progress
  - Purpose: detailed per-level and per-module progression.

- get_student_attempts
  - Endpoint: /api/v1/enrollment/{studentId}/attempts
  - Purpose: assessment attempt history and performance.

- get_subscription_state
  - Endpoint: /api/v1/subscription/me
  - Purpose: entitlement and active track validation.

## Guardrails for Correctness

- Never claim full lesson list or full progress unless live enrollment and progress calls succeed.
- Never say based on your profile unless profile tool was called in the same run.
- If live calls fail, return degraded mode output:
  - What is confirmed
  - What is not confirmed
  - Exact next check AI will perform when data is reachable
- Track-scope decisions must prioritize live subscription and enrollment over onboarding fallback.

## Opportunities and Job Board Personalization Rules

- Location precedence:
  1.  onboarding residentCountry and workCountry
  2.  user preferences location
  3.  explicit user-provided location in conversation
  4.  remote fallback
- Hard filter opportunities by location compatibility unless role is remote.
- Attach reason tags to each recommendation:
  - matched_track
  - matched_location
  - matched_skill_signals
  - matched_experience_level
- Reject or down-rank results where location is contradictory to user constraints.

## Caching and Freshness Policy

- Mirrored content cache: moderate TTL, sync by schedule and content change triggers.
- User state cache: very short TTL (30 to 120 seconds) or no cache for critical checks.
- Enrollment and progress data should not use long TTL for chat-critical decisions.
- Invalidate user-specific cache on:
  - progress updates
  - assessment submissions
  - subscription changes
  - onboarding updates that affect personalization

## Observability and Auditability

Log each AI response with data evidence metadata:

- profile_fetched_at
- enrollment_fetched_at
- progress_fetched_at
- subscription_fetched_at
- content_snapshot_version
- personalization_reason_tags

Add a confidence marker:

- high: all required live calls succeeded
- medium: partial live data with fallback
- low: live data unavailable

## Edge Cases to Cover

- Student has active subscription but no enrollment rows yet.
- Student has enrollment but no onboarding completion.
- Student has multiple tracks or changing track state.
- Course content exists locally but lesson lock state changed in LMS.
- Location missing or country code not normalized.
- Remote-only roles versus location-restricted roles.
- Assessment data delayed or partially available.

## Rollout Plan

1. Introduce missing live tools for enrollment, progress, attempts, and subscription.
2. Wire tool usage into planning prompts with strict must-call conditions.
3. Reduce user-state TTLs and add event-driven invalidation.
4. Add response evidence logging and confidence markers.
5. Enable location hard filtering in opportunities and job-board matching.
6. Run real-user data tests before marking fixes complete.

## Acceptance Criteria

- AI can correctly list accessible modules and lessons for current student state.
- AI can correctly report progress and assessment status with no stale claims.
- Opportunities and jobs respect track and location constraints.
- No claim of complete access or progress is made without live verification.
- Real-data test suite passes for all critical user journeys.

## Real-Data Test Matrix

- Student A: active track, partial progress, pending assessments.
- Student B: active track, completed level, certificate generated.
- Student C: AI mentor only, no course enrollment.
- Student D: London location, ensure no Toronto-only jobs unless remote.
- Student E: recent lesson progress update, verify immediate AI reflection.

## Notes for Future Implementation

- Keep this document as the baseline contract between frontend routes, AI tools, and LMS endpoints.
- Any new frontend route that shows user progression must map to a live AI tool endpoint.
- Any AI claim displayed to users should be traceable to a concrete endpoint fetch.

## Production Incident Addendum

### Incident Snapshot

Observed user-facing AI output:

- "I can't pull the full ordered list of all lessons in Module 3 because the course content isn't fully synced in my search system."

Why this is not acceptable:

- The student experience depends on AI being able to return ordered module and lesson views from authoritative data.
- Telling the user to check dashboard directly increases back and forth and reduces trust.

### Root Cause Class

- AI attempted to answer a structured curriculum question from content retrieval only.
- It did not use live enrollment structure endpoints as the source for ordered lesson visibility and lock state.

### Mandatory Rule for Ordered Lesson Queries

For any request containing terms like module, lesson list, full list, ordered list, what can I access, continue learning:

1. Call live structure tool first:
   - /api/v1/enrollment/{courseId}/structure
2. If progress detail is requested, also call:
   - /api/v1/enrollment/{courseId}/progress
3. Use mirrored content DB only for explanation and lesson content details, not access ordering truth.

### Response Contract for This Scenario

- If live structure call succeeds:
  - Return exact ordered lessons and lock/completion state.
- If live structure call fails:
  - Do not claim missing sync.
  - Return degraded mode with explicit status:
    - "I could not verify your live lesson structure right now."
    - "I can still explain SQL joins while I retry your structure data."

### Additional Acceptance Checks

- AI must return lesson ordering matching LMS structure output for the same course and level.
- AI must never infer full module completeness from content search index alone.
- AI must not use "content not fully synced" as a fallback phrase for enrollment structure failures.
