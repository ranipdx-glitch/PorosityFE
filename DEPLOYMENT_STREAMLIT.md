# Deploying PorosityFE to Streamlit Community Cloud

The browser front end lives in `app.py` at the repo root. Streamlit itself is
an **optional** dependency (declared as the `web` extra in `pyproject.toml`),
so it is **not** in the core `requirements.txt` — the CLI and library do not
need it. To run the web app you must install the `web` extra explicitly via
either `pip install -e .[web]` or `pip install -r requirements-web.txt`. The
instructions below mirror the WrinkleFE deployment process.

## 1. Verify locally

```bash
pip install -r requirements.txt
pip install -r requirements-web.txt   # streamlit (web extra)
pip install -e .
streamlit run app.py
```

Equivalently, you can install the core + web extras in a single step:

```bash
pip install -e .[web]
streamlit run app.py
```

Open http://localhost:8501 and confirm the **Profile**, **Mesh**, **Results**,
and **Stress** tabs populate after pressing **Run analysis** in the sidebar.

## 2. Push to GitHub

Commit the branch (typically `main`) to `ranipdx-glitch/porosityfe`. Streamlit
Cloud can deploy either `main` or a feature branch.

## 3. Authenticate with Streamlit Cloud

Visit <https://share.streamlit.io>, sign in with GitHub, and authorise access
to the GitHub organisation that owns the repository.

## 4. Deploy

Click **Create app → Deploy a public app from GitHub** and enter:

- Repository: `ranipdx-glitch/porosityfe`
- Branch: `main` (or a feature branch)
- Main file: `app.py`
- App URL: choose a subdomain like `porosityfe.streamlit.app`

Under **Advanced settings**, pick Python 3.11. Click **Deploy** — Streamlit
will install `requirements.txt` and run `streamlit run app.py` automatically.

> **Note on the `web` extra.** Streamlit is declared only in the optional
> `web` extra (and `requirements-web.txt`), not in the core `requirements.txt`,
> so CLI and library users do not pay for it. Streamlit Community Cloud's
> base image ships `streamlit` pre-installed, so the deploy above works
> as-is. For any other host (self-hosted, Docker, generic PaaS), install
> the `web` extra explicitly alongside `requirements.txt`:
> `pip install -e .[web]` or `pip install -r requirements-web.txt`.

## 5. Iterate

Pushes to the configured branch redeploy automatically. Use the gear icon to
view logs, change the Python version, or set secrets.

## Resource notes

The Community tier provides 1 GB RAM and 1 vCPU. The FE solve memory scales
roughly with `nx * ny * nz`; the default 30×10×12 mesh fits comfortably, but
expert-mode resolutions can exceed 1 GB. Reduce the mesh if you hit OOM
errors in the deploy logs.
