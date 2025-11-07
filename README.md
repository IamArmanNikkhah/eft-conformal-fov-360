# 🎥 PEFT-Conformal FoV Prediction for 360° Video Streaming

> University of Texas at Dallas — Immersive Media Systems Lab  

---

## 🧠 Overview

This project explores how **personalized multimodal Transformers** and **conformal prediction (CP)** can improve **360° video streaming** under unstable networks.

You will build a small simulator that:
1. Predicts a user’s **future field of view (FoV)** using a lightweight Transformer.
2. Personalizes that model per user using **PEFT (LoRA/adapter)** fine-tuning.
3. Produces **set-valued FoV regions** via **split conformal prediction**, ensuring a bounded miss-rate (≤ α).
4. Adapts α online through a feedback controller to trade off **risk vs. bandwidth**.

Your system will be evaluated by **Quality-of-Experience (QoE)** metrics such as viewport hit ratio, rebuffer ratio, and VWS-PSNR/VMAF-360 quality.

---

## 🧩 Repository Structure

```

peft-conformal-fov/
├── src/                # all source code
│   ├── geometry.py     # yaw/pitch ↔ vector, geodesic distance
│   ├── dataset.py      # data loaders (AVTrack360, Deep360Pilot)
│   ├── erp_grid.py     # ERP tiling utilities (6x12 default)
│   ├── player_stub.py  # simulation skeleton
│   └── ...
├── tests/              # pytest unit tests
├── scripts/            # runnable scripts (e.g., prep_data.py, run_sim.py)
├── configs/            # YAML/JSON config files
├── data/               # local datasets (ignored by Git)
│   └── .gitkeep
├── notebooks/          # Jupyter notebooks for quick experiments
├── env.yml             # Conda environment
├── requirements.txt    # pip alternative (optional)
├── .gitignore
├── README.md
└── LICENSE

````

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/peft-conformal-fov.git
cd peft-conformal-fov
````

### 2. Create and Activate Environment (Conda)

```bash
conda env create -f env.yml
conda activate fovenv
```

or, using pip:

```bash
python -m venv venv
source venv/bin/activate        # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
pytest
```

You should see:

```
collected 1 item
tests/test_basic.py .                                         [100%]
```

---

## 📦 Datasets

We’ll use publicly available 360° head-motion datasets.
Download them manually (they’re large — don’t push to GitHub!):

| Dataset          | Description                                           | Link                                                                 |
| ---------------- | ----------------------------------------------------- | -------------------------------------------------------------------- |
| **AVTrack360**   | Real head movements of users watching 360° videos     | [AVTrack360 dataset](https://github.com/AndreyTrekhleb/AVTrack360)   |
| **Deep360Pilot** | Trajectories for “piloting” tasks in immersive videos | [Deep360Pilot dataset](https://github.com/deep360pilot/deep360pilot) |

After downloading, place the raw files in:

```
data/AVTrack360/
data/Deep360Pilot/
```

and update their paths in your local `.env` or config file.

---

## 🚀 Quick Start (After Setup)

1. **Run the player stub** to inspect data flow:

   ```bash
   python -m src.player_stub --user 1 --seconds 30
   ```

   → outputs a CSV log of head orientation and tile indices.

2. **Train pooled Transformer (Week 2):**

   ```bash
   python scripts/train_pooled.py
   ```

3. **Fine-tune per-user adapters (Week 3):**

   ```bash
   python scripts/tune_peft.py --user 5
   ```

4. **Calibrate Conformal Prediction (Week 4):**

   ```bash
   python scripts/calibrate_cp.py
   ```

5. **Run full simulation with α-controller (Week 5):**

   ```bash
   python scripts/run_sim.py --config configs/default.yaml
   ```

---

## 🧪 Testing

Run all unit tests anytime:

```bash
pytest -v
```

Each new module should include small, isolated tests in `tests/`.

For code style checks (if pre-commit installed):

```bash
pre-commit run --all-files
```

---

## 📊 Evaluation Metrics

During final evaluation (Week 6), you’ll log:

* **Viewport Hit Ratio** – tiles within viewport correctly prefetched.
* **VWS-PSNR / VMAF-360** – perceptual quality inside the viewport.
* **Rebuffer Ratio** – stall time ÷ total play time.
* **Miss-Rate α̂** – fraction of frames where actual FoV lies outside CP set.

All results are aggregated across users and bandwidth conditions.

---

## 👩‍💻 Team Roles (Recommended)

| Role                     | Focus                                            |
| ------------------------ | ------------------------------------------------ |
| **Lead / Integrator**    | Repository hygiene, CI, experiment orchestration |
| **Model Lead**           | Transformer + PEFT tuning                        |
| **Personalization Lead** | Adapter integration, per-user pipelines          |
| **Calibration Lead**     | Conformal prediction and α-control               |
| **Systems Lead**         | Tiling, QoE metrics, and network emulation       |

Rotate tasks weekly so everyone touches both ML and systems code.

---

## 🗓️ Weekly Milestones

| Week | Focus                               | Key Deliverable              |
| ---- | ----------------------------------- | ---------------------------- |
| 1    | Environment, repo, data, geometry   | Working loader + player stub |
| 2    | Base Transformer + baselines        | Trained pooled model         |
| 3    | PEFT personalization                | Per-user adapters            |
| 4    | Conformal sets + tile mapping       | Validated coverage control   |
| 5    | Online α-controller + streaming sim | QoE comparisons              |
| 6    | Evaluation + presentation           | Final report + demo          |

---

## 🧰 Tech Stack

* **Language:** Python 3.11
* **Core Libraries:** PyTorch, NumPy, pandas, SciPy, matplotlib, tqdm, PEFT
* **Version Control:** Git + GitHub
* **Testing:** pytest
* **Optional:** W&B or MLflow for experiment tracking
* **Video Metrics:** VMAF-360 or VWS-PSNR tools

---

## 💡 Tips for Success

* Run small tests often — each module should work in isolation.
* Don’t push dataset files; use `.gitignore` wisely.
* Document parameters and choices in `configs/` for reproducibility.
* Keep your commits small and descriptive (e.g., “Add geodesic distance function”).
* Sync with your team daily — this project builds like Lego; each block depends on the last.


---

## 📜 License

MIT License — free to use and modify with attribution.

---

## 📨 Contact

Questions or issues?
Open an [issue](https://github.com/<your-username>/peft-conformal-fov/issues) or contact your course mentor.

---

*Happy coding, and remember: start simple, test everything, and commit early!* ✨

```

---

### 💬 Mentor’s Note
This README does three things at once:
1. **Educates:** briefly explains the purpose and context.  
2. **Guides:** provides clear setup and usage steps.  
3. **Aligns:** shows milestones and responsibilities.

Once you commit this to `README.md` on GitHub, your repo will instantly look professional and be usable by any new team member or reviewer.
```
