<h1>Non-Deterministic Unsupervised Models for Data Generation (MVTec AD)</h1>

<p>
This repo contains two Jupyter notebooks that implement and compare four autoencoder-style models with increasing stochasticity on the MVTec AD dataset:
</p>
<ul>
  <li><strong>DAE</strong> — Deterministic Autoencoder (baseline)</li>
  <li><strong>β-VAE</strong> — VAE with β-scaled KL  (baseline)</li>
  <li><strong>SAVAE</strong> — Mixture prior + heteroscedastic decoder + MC-dropout</li>
  <li><strong>AR-SAVAE</strong> — Scene latent + AR slot prior + LPIPS perceptual term</li>
</ul>

<p>
Target categories: <code>metal_nut</code> (primary) and <code>bottle</code> (sensitivity check).<br/>
Metrics: reconstruction error, <strong>FID</strong>, <strong>IS</strong>, plus uncertainty maps for SAVAE/AR-SAVAE.
</p>

<hr/>

<h2>Repo Layout</h2>

<pre><code>.
├─ notebooks/
│  ├─ mvtec_metal_nut.ipynb   <!-- training + eval on metal_nut -->
│  └─ mvtec_bottle.ipynb      <!-- same pipeline on bottle -->
└─ outputs/                   <!-- created at runtime (checkpoints, images, CSVs) -->
</code></pre>

<p><em>Note:</em> All code (data module, models, training &amp; eval helpers) lives inside the notebooks. There is no <code>src/</code> package.</p>

<hr/>

<h2>Dataset</h2>

<p>
We use the MVTec AD dataset. You can download the Kaggle mirror here:
<br/>
<a href="https://www.kaggle.com/datasets/ipythonx/mvtec-ad">https://www.kaggle.com/datasets/ipythonx/mvtec-ad</a>
</p>

<p>Expected folder layout (example for <code>metal_nut</code>):</p>
<pre><code>data/mvtec_ad/metal_nut/train/good/*.png
data/mvtec_ad/metal_nut/test/good/*.png
data/mvtec_ad/metal_nut/test/&lt;defect&gt;/*.png
data/mvtec_ad/metal_nut/ground_truth/&lt;defect&gt;/*.png
</code></pre>

<p>
Update the notebook paths if your dataset is elsewhere (e.g., Kaggle datasets mount).
</p>

<hr/>

<h2>Requirements</h2>

<ul>
  <li><strong>Python</strong> 3.10–3.11</li>
  <li><strong>GPU</strong> recommended (e.g., NVIDIA T4); CPU works but is slow</li>
</ul>

<p>Install (locally):</p>
<pre><code>pip install torch torchvision pytorch-lightning albumentations opencv-python \
            numpy pandas scipy lpips torch-fidelity matplotlib Pillow tqdm
</code></pre>

<p>Minimal (Kaggle/Colab often preinstalls many deps):</p>
<pre><code>pip install albumentations lpips torch-fidelity --quiet
</code></pre>

<hr/>

<h2>How to Run</h2>

<ol>
  <li>Place the dataset under <code>data/mvtec_ad/</code> (or mount Kaggle dataset).</li>
  <li>Open <code>notebooks/mvtec_metal_nut.ipynb</code> (or <code>mvtec_bottle.ipynb</code>).</li>
  <li>Run cells top-to-bottom:
    <ul>
      <li><em>Data &amp; transforms</em> → builds train/val/test splits</li>
      <li><em>Models</em> → DAE / β-VAE / SAVAE / AR-SAVAE</li>
      <li><em>Training</em> → checkpoints + recon/sample grids saved to <code>outputs/</code></li>
      <li><em>Evaluation</em> → FID/IS via <code>torch-fidelity</code>, summary CSV</li>
    </ul>
  </li>
</ol>

<p><strong>Outputs</strong> (created automatically):</p>
<ul>
  <li><code>outputs/&lt;model&gt;/</code> — checkpoints, metrics CSVs</li>
  <li><code>outputs/*_val_recons.png</code> — recon grids</li>
  <li><code>outputs/*_samples.png</code> — prior samples</li>
  <li><code>outputs/eval/&lt;model&gt;/gen/</code> — generated images for FID/IS</li>
  <li><code>outputs/gen_eval_summary.csv</code> — FID/IS summary table</li>
</ul>

<hr/>

<h2>Notes &amp; Tips</h2>

<ul>
  <li><strong>LPIPS</strong> (AR-SAVAE) will fetch VGG weights on first use (internet required once).</li>
  <li><strong>FID variance</strong> can be high with a small reference set (e.g., only 22 “good” images).
      Compare models within the <em>same category</em> under the <em>same protocol</em>.</li>
  <li><strong>Repro</strong>: we seed PyTorch/NumPy, sort keys, and use seeded splits in the notebooks.</li>
  <li>If Lightning warns about existing checkpoint dirs, they will be reused; delete older <code>outputs/&lt;model&gt;</code> to start fresh.</li>
</ul>

<hr/>

<h2>Typical Results (single-run examples)</h2>

<ul>
  <li><code>metal_nut</code>: AR-SAVAE best run around FID ≈ 250, IS ≈ 1.09</li>
  <li><code>bottle</code>: AR-SAVAE ≈ FID 330.42, IS ≈ 1.064 (category sensitivity)</li>
</ul>

<p>
Absolute FID values shouldn’t be compared across categories since texture statistics and small reference sizes affect the score.
</p>

<hr/>

<h2>Troubleshooting</h2>

<ul>
  <li><em>AttributeError: missing layer</em> — ensure you ran the full model-definition cell <em>after</em> any edits.</li>
  <li><em>FID/IS very high</em> — increase generated sample count (e.g., 1000–5000), ensure <code>model.eval()</code> for evaluation.</li>
  <li><em>cuDNN/cuBLAS “already registered”</em> — benign on some notebook runtimes; usually safe to ignore.</li>
</ul>

<hr/>

<h2>License</h2>
<p>
Academic/research use for the evaluated MVTec AD subsets. See MVTec’s license for dataset terms.
</p>
