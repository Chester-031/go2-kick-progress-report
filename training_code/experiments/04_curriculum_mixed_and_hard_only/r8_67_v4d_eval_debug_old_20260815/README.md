One-step evaluation boundary debug. It reuses the metric-only patch, disables
video, and prints markers around reset, policy inference, low_env.step, and
end_high_action. A 180-second timeout prevents another long GPU stall.
