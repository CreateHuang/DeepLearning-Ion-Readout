import torch
import os
import numpy as np


pth_path = r"F:\MobileUNET_master\outputs\train_unet\2-best.pth"
save_dir = r"F:\MobileUNET_master\model_weights"


def save_all_parameters(pth_path, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    print(f"保存目录已确认：{save_dir}")

    try:
        state_dict = torch.load(pth_path, map_location=torch.device("cpu"))
        print(f"成功加载权重文件，共包含 {len(state_dict)} 个参数")
    except Exception as e:
        print(f"加载权重文件失败：{str(e)}")
        return

    for param_name, param_value in state_dict.items():

        safe_filename = param_name.replace(".", "_").replace("/", "_") + ".txt"
        save_path = os.path.join(save_dir, safe_filename)

        param_np = param_value.cpu().numpy()

        try:
            with open(save_path, "w", encoding="utf-8") as f:

                f.write(f"# 参数名称：{param_name}\n")
                f.write(f"# 参数形状：{param_np.shape}\n")
                f.write(f"# 参数数据类型：{param_np.dtype}\n")
                f.write("# 参数值（保留6位小数）：\n")

                if param_np.ndim == 0:
                    f.write(f"{param_np:.6f}\n")
                elif param_np.ndim == 1:
                    np.savetxt(f, param_np.reshape(-1, 1), fmt="%.6f")
                else:
                    np.savetxt(f, param_np.reshape(-1, param_np.shape[-1]), fmt="%.6f")

            print(f"已保存：{safe_filename}")
        except Exception as e:
            print(f"保存失败 {param_name}：{str(e)}")

    print(f"\n所有参数已保存至 {save_dir}，共 {len(state_dict)} 个文件")


if __name__ == "__main__":
    save_all_parameters(pth_path, save_dir)
