terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.5"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_artifact_registry_repository" "face_tracking" {
  location      = var.region
  repository_id = "face-tracking"
  format        = "DOCKER"
}

resource "google_cloud_run_v2_service" "face_tracking" {
  name     = "face-tracking"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    scaling {
      min_instance_count = 1
      max_instance_count = 20
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/face-tracking/app:latest"

      ports {
        container_port = 8000
      }

      env {
        name  = "MAX_CONCURRENT_DETECTIONS"
        value = "10"
      }

      env {
        name  = "MAX_UPLOAD_BYTES"
        value = "10485760"
      }

      env {
        name  = "TRIAGE_CONFIDENCE_THRESHOLD"
        value = "0.6"
      }

      env {
        name  = "RATE_LIMIT_PER_MINUTE"
        value = "30/minute"
      }

      env {
        name = "ANTHROPIC_API_KEY"
        value_source {
          secret_key_ref {
            secret  = "anthropic-api-key"
            version = "latest"
          }
        }
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "1Gi"
        }
      }
    }
  }
}
