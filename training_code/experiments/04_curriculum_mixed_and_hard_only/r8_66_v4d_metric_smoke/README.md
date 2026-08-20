R8.66 disables only the imageio/render path inside run_v4d_evaluation.
The simulator, policies, trajectory CSV and summary JSON remain active.
This is a 60-step smoke test before multi-seed evaluation. The original
video path is untouched and is used only after metrics pass.
