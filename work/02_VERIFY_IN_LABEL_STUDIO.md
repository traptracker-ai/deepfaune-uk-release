# Stage 2 — Verify in Label Studio (no notebook)

This stage happens in the Label Studio UI at http://localhost:8080, not a notebook.

1. Log in / create a local account (stored in ./labelstudio-data).
2. Create a new project.
3. **Settings → Labeling Interface**: paste the contents of
   `/dataset/ls_import/label_config.xml`
4. **Settings → Cloud Storage → Add Source → Local files**:
   - Absolute local path: `/label-studio/files`
   - tick "Treat every bucket object as a source file"
   - **Sync**. (This makes incoming/ images viewable. If images show as broken,
     this step is the cause.)
5. **Import** `/dataset/ls_import/tasks.json` — boxes + species appear as
   pre-annotations.
6. Review every task: approve correct predictions, fix boxes/labels otherwise,
   delete false boxes. Prioritise the low-confidence detections Stage 1 reported.
7. **Export → JSON** → save to `/dataset/ls_export/export.json`
8. Run Stage 3 (`03_merge.ipynb`).

Only tasks you SUBMIT are merged. Untouched tasks are ignored. Blank frames
(no boxes) are skipped by default in Stage 3.
