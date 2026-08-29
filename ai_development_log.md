# AI Development Log

**Project:** Loan Performance Intelligence Engine
**Challenge:** Intain Campus FinTech Challenge 2026 — AI Track

This log documents all AI tool usage throughout the development of this project, as required by Task 8 of the problem statement.

---

## Session 1 — Project Scaffold (2026-08-29)

### AI Tool Used
- **Model:** Google Gemini (Antigravity IDE agent)
- **Purpose:** Project scaffolding, directory structure, README, configuration

### Representative Prompt
```
Build a non-LLM-first ML system for the Loan Performance Intelligence Engine.
Start with Phase 0: Repo structure, README, requirements.txt, synthetic-data generator
matching the exact schemas from the problem statement.
```

### What Was Accepted
- Directory structure following modular ML project conventions
- .gitignore covering Python, data files, model checkpoints
- README template with setup instructions and reproducibility notes
- CHANGELOG format
- Configuration module with centralized paths and random seeds

### What Was Rejected / Modified
- (None yet — first commit is standard scaffold)

### Human Review Process
- Reviewed implementation plan before approving code generation
- Verified directory structure covers all 7 required tasks
- Confirmed .gitignore won't accidentally include large data files

### Approximate AI-Generated Code Share
- **This commit:** ~95% AI-generated (scaffold is boilerplate)
- Human contribution: problem statement analysis, architectural decisions, approval

### Lessons Learned
- Starting with a detailed implementation plan before any code prevents wasted commits
- Pinning dependency versions from the start avoids reproducibility issues later
