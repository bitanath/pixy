Here's the complete plan:
2. Update build.zig — fixed-target artifacts
Replace the single native-target module with two explicit targets:
Module	Target
macos_mod	.cpu_arch = .aarch64, .os_tag = .macos
linux_mod	.cpu_arch = .aarch64, .os_tag = .linux, .abi = .musl
watchos_mod	(existing)
- Remove b.standardTargetOptions — no more native-target ambiguity
- Remove run step (cross-compiled exes can't run on host)
- All 3 targets build from core/index.zig (lib) and core/main.zig (exe)
- Linux uses musl for fully static binaries (no glibc dependency on the target)
3. Update framework/scripts/deploy.sh
- zig build builds all 3 targets (macOS + Linux + watchOS simultaneously)
- Copy llm-watchos.a → watch app (unchanged)
- Compile test with outputs/lib/llm-macos.a instead of outputs/lib/llm.a
- Copy outputs/bin/llm-linux to a known location (project root? or performix/?) — I need your preference on this
4. Clean up stale files
rm outputs/bin/llm          # replaced by llm-macos
rm outputs/lib/llm.a        # replaced by llm-macos.a
rm outputs/lib/llm-linux.a  # regenerated
rm outputs/lib/llm-macos.a  # regenerated
After the new build.zig, the outputs/ tree will cleanly contain:
outputs/bin/
  llm-macos      ← macOS exe (used directly by user)
  llm-linux      ← Linux exe (used for performix)
outputs/lib/
  llm-macos.a    ← macOS static lib (linked by test.swift)
  llm-linux.a    ← Linux static lib  (used later )
  llm-watchos.a  ← watchOS static lib (copied to watch app)
The old outputs/bin/llm and outputs/lib/llm.a are vestiges that need removal.
5. Order of execution

2. build.zig — fixed targets, strip, renames
3. deploy.sh — update lib path, optional linux deployment
4. Delete benchmark.swift, clean up stale artifacts
5. Build + test to verify everything works end-to-end
