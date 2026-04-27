# First Claude Code session — kickoff prompt

Copy the prompt below into your first Claude Code session. It tells Claude
Code what to read, what to do, and what the acceptance criteria are.

---

## Paste this into Claude Code:

```
Please read the following files in order before doing anything else:
1. README.md — project overview
2. CLAUDE.md — how to work on this project
3. NEXT_STEPS.md — current task and acceptance criteria
4. PARSER_DESIGN.md — technical reference for the parser

Then briefly summarise back to me:
- What this project is and its current state
- What the active task is
- What the acceptance criteria are for that task
- Any questions or concerns before you start

After I confirm, proceed with the active task. Follow the workflow in
CLAUDE.md: inspect the document first before writing any parser code,
ask before adding dependencies, and commit in small increments.

The PDF for the current task should be placed at
docs/Building_Regulations_2018.pdf. If it's not there yet, tell me and I'll
download it.
```

---

## Why this prompt works

- It forces Claude Code to load context from your files rather than guessing
- The "summarise back" step catches misunderstandings before any code gets
  written — if it summarises wrong, you correct it cheaply
- It references the workflow rules (CLAUDE.md) so you don't have to
  re-specify them every session
- It creates a natural checkpoint before work starts

## Subsequent sessions

For later sessions, the prompt can be much shorter:

```
Read CLAUDE.md and NEXT_STEPS.md. Continue the active task. Tell me what
you're about to do before you do it.
```

## If you get stuck mid-task

If Claude Code goes down a wrong path, the fastest reset is:

```
Stop. Let's back up. What does NEXT_STEPS.md say the acceptance criteria
are? What have you done so far, and which of those criteria are met?
```

This forces a self-assessment and usually surfaces the misalignment.

## A realistic expectation

Your first session on the Regulations parser will probably be 60-90 minutes
and will not finish the task. That's fine. Target for session one:
document inspection done, layout diagnosed, a skeleton parser running on
the first Part of the Regulations with a handful of test chunks extracted.
Session two: finish Schedule 3. Session three: cleanup and acceptance
criteria pass.

If the first session takes much longer than that, something is going
sideways — stop and ask me (in web chat) to help diagnose before pushing
forward.
