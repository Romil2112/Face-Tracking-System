output "artifact_registry_url" {
  description = "Docker repository base URL for face-tracking images"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.face_tracking.repository_id}"
}

output "cloud_run_url" {
  description = "Deployed Cloud Run service URL"
  value       = google_cloud_run_v2_service.face_tracking.uri
}
