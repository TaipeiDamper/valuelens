import os

file_path = "valuelens/ui/overlay_window.py"
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 我們要刪除原本 refresh_frame 殘留下來的舊邏輯
# 根據最近一次 view_file，冗餘代碼大約在 866 到 882 行 (0-indexed 是 865 到 881)
# 我們精確匹配字串來刪除，避免行號跑掉

new_lines = []
skip = False
for line in lines:
    if "rect = self._lens_rect()" in line and "_on_capture_finished" in "".join(lines[lines.index(line)-20:lines.index(line)]):
        skip = True
        print(f"Starting skip at: {line.strip()}")
    
    if not skip:
        new_lines.append(line)
        
    if skip and "is_changed, mae = self.scene_detector.detect_change(gray_frame)" in line:
        skip = False
        new_lines.append(line) # 保留這一行，它是我們要的
        print(f"Ending skip at: {line.strip()}")

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Fix completed.")
