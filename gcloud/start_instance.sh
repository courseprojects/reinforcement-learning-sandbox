gcloud compute \
--project "cs221-project-293411" \
instances create "thor" \
--zone "europe-west1-d" \
--machine-type "custom-10-61440" \
--subnet "default" \
--maintenance-policy "TERMINATE" \
--service-account "760987773404-compute@developer.gserviceaccount.com" \
--scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring.write","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" \
--accelerator type=nvidia-tesla-p100,count=1 \
--min-cpu-platform "Intel Broadwell" \
--image "nvidia-gpu-cloud-image-20200629" \
--image-project "nvidia-ngc-public" \
--boot-disk-size "32" \
--boot-disk-type "pd-standard" \
--boot-disk-device-name "thor"  \
--metadata-from-file startup-script=./setup.sh