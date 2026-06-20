import os
import shutil

src_dir = r"C:\Users\Acer\.gemini\antigravity-ide\brain\69fa48f9-1bf9-487d-bd42-69177261f162"
dest_dir = r"g:\Jivesh data transfer\Projects\Audio_emotions\images"

os.makedirs(dest_dir, exist_ok=True)

# Copy screenshots
shutil.copy(os.path.join(src_dir, "media__1781936694835.png"), os.path.join(dest_dir, "light_mode.png"))
shutil.copy(os.path.join(src_dir, "media__1781936729693.png"), os.path.join(dest_dir, "dark_mode.png"))

print("Screenshots copied successfully to g:\\Jivesh data transfer\\Projects\\Audio_emotions\\images!")
