#!/bin/bash

# Exit on error
set -e

# Check if commit hash is passed as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <commit-hash>"
  exit 1
fi

COMMIT_HASH=$1
RELEASES_DIR="/home/ubuntu/releases"
DEPLOY_DIR="/home/ubuntu/aegra"
SERVICE_NAME="aegra"
RELEASE_NAME="aegra-${COMMIT_HASH}"
declare -a INSTANCES=("0" "1" "2")

# Check if the release directory exists
if [ ! -d "${RELEASES_DIR}/${RELEASE_NAME}" ]; then
  echo "Release ${RELEASE_NAME} not found in ${RELEASES_DIR}"
  exit 1
fi

# Keep a reference to the previous deployment
if [ -L "${DEPLOY_DIR}" ] || [ -d "${DEPLOY_DIR}" ]; then
  PREVIOUS="${DEPLOY_DIR}.backup"
  echo "Backing up current deployment to ${PREVIOUS}..."
  rm -rf "${PREVIOUS}"
  cp -r "${DEPLOY_DIR}" "${PREVIOUS}"
else
  echo "No previous deployment found, first deployment in progress."
  PREVIOUS=""
fi

rollback_deployment() {
  if [ -n "$PREVIOUS" ] && [ -d "$PREVIOUS" ]; then
    echo "Rolling back to previous deployment: ${PREVIOUS}"
    rm -rf "${DEPLOY_DIR}"
    cp -r "${PREVIOUS}" "${DEPLOY_DIR}"
  else
    echo "No previous deployment to roll back to."
    return 1
  fi

  # Wait before restarting services
  sleep 10

  # Restart all service instances with the previous code
  for instance in "${INSTANCES[@]}"; do
    SERVICE="${SERVICE_NAME}@${instance}.service"
    echo "Restarting $SERVICE..."
    sudo systemctl restart "$SERVICE"
  done

  echo "Rollback completed."
}

# Deploy the new release
echo "Promoting ${RELEASE_NAME} to ${DEPLOY_DIR}..."
rm -rf "${DEPLOY_DIR}"
cp -r "${RELEASES_DIR}/${RELEASE_NAME}" "${DEPLOY_DIR}"

WAIT_TIME=5
restart_service() {
  local instance=$1
  local SERVICE="${SERVICE_NAME}@${instance}.service"
  echo "Restarting ${SERVICE}..."

  # Restart the service
  if ! sudo systemctl restart "$SERVICE"; then
    echo "Error: Failed to restart ${SERVICE}. Rolling back deployment."
    rollback_deployment
    exit 1
  fi

  # Wait a few seconds to allow the service to fully start
  echo "Waiting for ${SERVICE} to fully start..."
  sleep $WAIT_TIME

  # Check the status of the service
  if ! systemctl is-active --quiet "${SERVICE}"; then
    echo "Error: ${SERVICE} failed to start correctly. Rolling back deployment."
    rollback_deployment
    exit 1
  fi

  echo "${SERVICE} restarted successfully."
}

# Restart all service instances
for instance in "${INSTANCES[@]}"; do
  restart_service "$instance"
done

echo "Deployment of ${RELEASE_NAME} completed successfully."
echo "Previous deployment backed up at: ${PREVIOUS}"
