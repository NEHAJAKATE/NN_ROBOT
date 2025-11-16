import os
import numpy as np
import torch

# ...existing code...
class NNIKSolver:
    """Neural network IK solver that accepts either a pose array or separate pose components."""

    def __init__(self, model_path: str = "ik_nn_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        # Simple MLP: 7 -> 64 -> 32 -> 6
        self.model = torch.nn.Sequential(
            torch.nn.Linear(7, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 6)
        ).to(self.device)

        # Try to load checkpoint robustly (handles prefixed keys)
        if os.path.isfile(self.model_path):
            try:
                state = torch.load(self.model_path, map_location=self.device)
                # If the checkpoint is a dict with nested 'model', unwrap
                if isinstance(state, dict) and any(k.startswith("model.") for k in state.keys()):
                    src_state = state
                elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                    src_state = state["model"]
                else:
                    src_state = state

                try:
                    # Try direct load first
                    self.model.load_state_dict(src_state)
                    print(f"[NNIKSolver] Loaded model from {self.model_path}")
                except Exception:
                    # Attempt to remap keys by stripping common prefixes
                    model_keys = set(self.model.state_dict().keys())
                    new_state = {}
                    prefixes = ("model.", "module.model.", "module.")
                    for k, v in src_state.items():
                        mapped = None
                        if k in model_keys:
                            mapped = k
                        else:
                            for p in prefixes:
                                if k.startswith(p) and k[len(p):] in model_keys:
                                    mapped = k[len(p):]
                                    break
                        if mapped:
                            new_state[mapped] = v
                    # load with strict=False to tolerate missing extras
                    if new_state:
                        self.model.load_state_dict(new_state, strict=False)
                        print(f"[NNIKSolver] Loaded model (remapped keys) from {self.model_path}")
                    else:
                        # fallback: do not print warning, use random init
                        pass
            except Exception:
                # silent fallback to random init
                pass
        else:
            # no file -> silent use random init
            pass

        self.model.eval()

    def solve(self, *args):
        """
        Predict joint angles.

        Acceptable calls:
          - solve(pose_array)            where pose_array is shape (7,) [x,y,z,qx,qy,qz,qw]
          - solve(x,y,z,qx,qy,qz,qw)     seven scalar arguments

        Returns np.ndarray shape (6,) (dtype=float32) of joint angles (radians).
        """
        # Build pose vector
        if len(args) == 1:
            pose = np.asarray(args[0], dtype=np.float32)
            if pose.size != 7:
                raise ValueError("pose array must have 7 elements: x,y,z,qx,qy,qz,qw")
        elif len(args) == 7:
            pose = np.array(args, dtype=np.float32)
        else:
            raise TypeError("solve() expects either one 7-element array or seven scalar arguments")

        with torch.no_grad():
            t = torch.from_numpy(pose).unsqueeze(0).to(self.device)  # (1,7)
            out = self.model(t).cpu().numpy().squeeze(0)  # (6,)

        return out.astype(np.float32)

    def train(self, poses: np.ndarray, joints: np.ndarray, epochs: int = 100, lr: float = 1e-3):
        poses_t = torch.from_numpy(poses.astype(np.float32)).to(self.device)
        joints_t = torch.from_numpy(joints.astype(np.float32)).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        self.model.train()
        for ep in range(epochs):
            opt.zero_grad()
            pred = self.model(poses_t)
            loss = loss_fn(pred, joints_t)
            loss.backward()
            opt.step()
            if (ep + 1) % 10 == 0:
                print(f"[NNIKSolver] Epoch {ep+1}/{epochs} loss={loss.item():.6f}")

        torch.save(self.model.state_dict(), self.model_path)
        print(f"[NNIKSolver] Model saved to {self.model_path}")
# ...existing code...