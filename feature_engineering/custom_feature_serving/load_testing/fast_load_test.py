import os
import json
from datetime import datetime, timedelta
from typing import Tuple

import requests
from locust import FastHttpUser, task, events, constant_throughput
from locust.runners import WorkerRunner, STATE_STOPPING, STATE_STOPPED, STATE_CLEANUP
import locust.runners

# Trigger CPU warning only at very high utilization
locust.runners.CPU_WARNING_THRESHOLD = 98.0


class LoadTestUser(FastHttpUser):    
    def get_oauth_token(self, token_lifetime : timedelta = timedelta(minutes=55)) -> Tuple[str, datetime]:
        """
        This function is used to get the oauth token and returns the token as a string and the expiration datetime
        """
        # Make the request to get the access token
        response = requests.post(
            url=f'{self.workspace_url}/oidc/v1/token',
            auth=(self.CLIENT_ID, self.CLIENT_SECRET),
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            data={
                'grant_type': 'client_credentials',
                'scope': 'all-apis',
                "authorization_details" : json.dumps([{
                    "type" :  "workspace_permission",
                    "object_type" : "serving-endpoints",
                    "object_path" :  f"/serving-endpoints/{self.endpoint_id}",
                    "actions" : ["query_inference_endpoint"]
                }])
            }
        )
        return response.json().get('access_token'), datetime.now() + token_lifetime
    
    def on_start(self):
        # Read the input json file from disk
        with open("input.json", "r") as json_features:
            self.model_input = json.load(json_features)
        # Load environment variables as necessary
        self.endpoint_name = os.environ.get("DATABRICKS_ENDPOINT_NAME")
        self.CLIENT_ID = os.environ.get("CLIENT_ID")
        self.CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
        self.workspace_url = os.environ.get("DATABRICKS_WORKSPACE_URL")
        self.endpoint_id = os.environ.get("ENDPOINT_ID")
        self.oauth, self.expiration = self.get_oauth_token()
        
    def check_token_expiration(self):
        """
        This function is used to check if the oauth token has expired and if it has, refreshes the token
        """
        if datetime.now() > self.expiration:
            self.oauth, self.expiration = self.get_oauth_token()

    # Define the request task
    @task
    def query_single_model(self):
        self.check_token_expiration()
        # Read the latest token from the environment variable, this allows oauth token to be updated
        headers = {"Authorization": f"Bearer {self.oauth}"}
        self.client.post(f"serving-endpoints/{self.endpoint_name}/invocations",
                         headers=headers,
                         json=self.model_input)
        

@events.quitting.add_listener
def on_quitting(environment, **_kwargs):
    # skip printing in workers
    if isinstance(environment.runner, WorkerRunner):
        return

    total = environment.stats.total
    summary = {
        "Total Requests": total.num_requests,
        "Total Failures": total.num_failures,
        "Min":   f"{total.min_response_time} ms",
        "P50":      f"{total.get_response_time_percentile(0.50)} ms",
        "P90":      f"{total.get_response_time_percentile(0.90)} ms",
        "P95":      f"{total.get_response_time_percentile(0.95)} ms",
        "P99":      f"{total.get_response_time_percentile(0.99)} ms",
        "P99.9":    f"{total.get_response_time_percentile(0.999)} ms",
    }
    print("=== Final Summary ===")
    for k, v in summary.items():
        print(f"{k:15}: {v}")
