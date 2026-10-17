# BrainTumorFL

A federated learning research demonstration for classifying brain MRI images into four categories using a shared global model.

The project demonstrates:
- Federated training with two clients and a central server using Flower/FedAvg.
- Client-side evaluation and reporting of accuracy to the server.
- Server-side aggregation and display of individual and average accuracies.
- Sharing the final global model and metrics over HTTP.
- A Flask dashboard where users upload an MRI image and view a predicted class, confidence, and class probabilities.

> **Research use only:** This project is a learning demonstration, not a medical diagnostic system. Do not use its predictions for clinical decisions.

## Classes

The model predicts one of these four dataset classes:
- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

## Repository layout

```text
BrainTumorFL/
├── client/
│   ├── client.py
│   └── predict.py
├── common/
│   ├── config.py
│   └── model.py
├── dashboard/
│   └── app.py
├── server/
│   ├── server.py
│   └── model_share.py
├── models/
│   ├── global_model.pth   # generated after training; not committed
│   └── metrics.json       # recorded federated metrics
├── requirements.txt
└── README.md
```

## Architecture

```text
   client 1                        client 2
(local dataset)                (local dataset)
      |                              |
      +---------- local training ----+
                     |
                     v
            Flower server (FedAvg)
                     |
       client evaluation metrics
                     |
                     v
             server's computer
       global model + metrics files
                     |
              HTTP service :8082
                 /model /metrics
                     |
             +-------+-------+
             v               v
client 1 dashboard      client 2 dashboard
    MRI upload UI          MRI upload UI
```

## Requirements

- Python and the dependencies listed in `requirements.txt`.
- Git.
- A local MRI dataset on each training client, arranged as expected by `client/client.py`.
- For a live demo, the computers must be able to reach Palak's Mac over the same network.

## Installation

Clone the repository:

```bash
git clone https://github.com/palakadsul/BrainTumorFL.git
cd BrainTumorFL
```

Create and activate a virtual environment.

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset setup

Keep each client's dataset on that client's computer. The current setup expects a directory containing `Training` and `Testing`, with one subfolder per class:

```text
dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Check `client/client.py` and `common/config.py` if your dataset layout differs.

## Run fresh federated training

Use separate terminals. Start the server first.

### 1. Start the Flower server 

From the project root:

```bash
python -m server.server
```

The current setup uses Flower address `0.0.0.0:8081`. Confirm the address in `server/server.py` if the configuration has changed. Keep this terminal running.

### 2. Start client 1

Activate the project environment on client 1, then run:

```bash
python -m client.client \
  --server 172.16.237.173:8081 \
  --data "/path/to/your/dataset" \
  --client-id name
```

Replace the example data path with the real dataset directory.

### 3. Start client 2

On Gauri's Windows computer:

```powershell
python -m client.client --server 172.16.237.173:8081 --data "D:\path\to\your\dataset" --client-id name
```

Replace the example path with the real dataset directory.

### 4. Observe the server

The server terminal should show the training rounds, client evaluation accuracies, average accuracy metrics, and model-saving messages. Exact output depends on the configuration and the current run.

## Run the model-sharing service

Once the trained global model and metrics are present on Palak's computer, start the HTTP service in a separate terminal:

```bash
python -m server.model_share
```

The current service uses port `8082`:

- `GET /health` — service status and whether model/metrics files exist.
- `GET /metrics` — saved federated metrics as JSON.
- `GET /model` — downloads the global model.

Test locally:

```bash
curl http://127.0.0.1:8082/health
curl http://127.0.0.1:8082/metrics
```

From another computer, use Palak's current IP address:

```bash
curl http://172.16.237.173:8082/health
curl http://172.16.237.173:8082/metrics
```

If server's IP changes, update `MODEL_SERVER` in `dashboard/app.py` on the shared codebase and push the change. The sharing service must stay running while teammates use their dashboards.

## Run the MRI upload dashboard

Start the dashboard in a separate terminal on each computer:

```bash
python -m dashboard.app
```

Open this URL in the browser on that same computer:

```text
http://127.0.0.1:5000
```

The dashboard retrieves current metrics from server's sharing service. On image submission, it downloads the global model and performs inference locally.

### Typical daily demo startup (reuse the trained model)

You do **not** need to retrain just to demonstrate prediction with an existing saved model.

1. On Palak's Mac, start `python -m server.model_share`.
2. Start `python -m dashboard.app` on each computer that needs a UI.
3. Confirm the dashboard says the server metrics were received.
4. Upload an MRI image and review the prediction.

For a fresh training demonstration, also start `python -m server.server` and both client commands above. After training, ensure the model and metrics served by the sharing service are the results of that run.

## Recorded results from the previous run

The completed three-round run reported these round 3 values:

| Metric | Recorded value |
|---|---:|
| client 1 evaluation accuracy | 99.10% |
| client 2 evaluation accuracy | 98.25% |
| Simple average accuracy | 98.67% |
| Sample-weighted average accuracy | 98.65% |

These values describe the recorded run; they are not guaranteed for future runs or unseen images. The dashboard should display the metrics currently returned by `/metrics`.

## Troubleshooting

### Port 5000 is already in use

On macOS, identify the process:

```bash
lsof -nP -iTCP:5000 -sTCP:LISTEN
```

Only stop the process if you have confirmed it is an old dashboard you own. Do not stop macOS system processes such as `ControlCenter`. If necessary, configure a different dashboard port in `dashboard/app.py` and use the matching local URL.

### Sharing service is unavailable

```bash
curl http://127.0.0.1:8082/health
```

Confirm `server.model_share` is running and that `models/global_model.pth` and `models/metrics.json` exist on server's computer.

### Dashboard cannot receive metrics

- Confirm the computers can reach one another over the network.
- Confirm server's current IP address.
- Check `MODEL_SERVER` in `dashboard/app.py`.
- Test `/health` and `/metrics` from the client computer.
- Check firewall settings if the service runs locally but is unreachable remotely.

### Global model is missing

Complete a training run or ensure the saved global model is available at the path expected by `server/model_share.py`. Check `/health` for `model_available: true`.

## Notes and limitations

- Client datasets remain on client computers; the configured federated workflow exchanges model updates and evaluation metrics.
- Displayed accuracy values are evaluation metrics reported by the server. Uploading a single MRI does not recalculate model accuracy.
- Model confidence is not the same as a clinically validated probability.
- The basic HTTP sharing service is intended for a controlled demo network, not an untrusted public network.
- Do not commit private data, datasets, credentials, or generated model files to Git.

## License

Add a `LICENSE` file if you intend to distribute or reuse this project publicly.
