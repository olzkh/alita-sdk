import time
import requests
from typing import Dict, Any


def handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    print("[INFO] Starting")
    try:
        action = event.get("action")
        print("Action:" + action)
        project_id = event.get("project_id")
        print("Id:" + project_id)
        carrier_url = event.get("url")
        carrier_token = event.get("token")
        change_in_percentages = event.get("change_in_percentages")
        print("Change %:" + change_in_percentages)
        env = event.get("env")
        print("Env:" + env)
        target = event.get("target")
        print("Target:" + target)
        aggregation = event.get("aggregation")
        print("Aggregation:" + aggregation)
        comparison = event.get("comparison")
        print("Comparison:" + comparison)
        test_modified = event.get("test_modified")
        print("Test:" + test_modified)
        url = carrier_url + '/api/v1/backend_performance/thresholds/' + project_id
        baseline_url = carrier_url + '/api/v1/backend_performance/baseline/' + project_id + '?test_name=' + test_modified + '&env=' + env
        headers = {
            'Authorization': f'Bearer {carrier_token}',
            'Accept': '*/*'
        }
        print(url)
        if action == 'create':
            baseline_requests = {}
            response_baseline = requests.get(baseline_url, headers=headers)
            time.sleep(3)
            data_baseline = response_baseline.json()
            max_threshold = 0.0
            for entry in data_baseline['baseline']:
                if entry[aggregation] > max_threshold:
                    max_threshold = entry[aggregation]
                new_value = round(
                    float(entry[aggregation]) / 100.0 * float(change_in_percentages) + float(entry[aggregation]), 1)
                baseline_requests[entry['request_name']] = {'test': entry['simulation'],
                                                            'environment': entry['env'],
                                                            'scope': entry['request_name'], 'target': target,
                                                            'aggregation': aggregation, 'comparison': comparison,
                                                            'value': new_value}
                create_response = requests.post(url + '/', headers=headers,
                                                json=baseline_requests[entry['request_name']])
                print("Created response: " + str(create_response.status_code))
                time.sleep(3)
            every_value = round(float(max_threshold) / 100.0 * float(change_in_percentages) + float(max_threshold),
                                1)
            every = {'test': test_modified, 'environment': env, 'scope': 'every', 'target': target,
                     'aggregation': aggregation, 'comparison': comparison, 'value': every_value}
            every_response = requests.post(url, headers=headers, json=every)
            return {'statusCode': 200, 'body': f"Thresholds created"}
        elif action == 'update':
            response = requests.get(url, headers=headers)
            modified = {}
            if response.status_code == 200:
                data = response.json()
                for entry in data['rows']:
                    if (entry['test'] == test_modified) and (entry['target'] == target):
                        new_entry = entry
                        new_entry['value'] = round(
                            float(entry['value']) / float(100.0) * float(change_in_percentages) + float(
                                entry['value']), 1)
                        new_entry.pop('project_id', None)
                        modified[entry['id']] = new_entry
                for entry in modified:
                    update_response = requests.put(url + '/' + str(entry), headers=headers, json=modified[entry])
                    print("Updated response:", update_response.status_code)
                    time.sleep(3)
            else:
                print('Failed to get data:', response.status_code, response.text)
            return {'statusCode': 200, 'body': f"Thresholds updated"}
        elif action == 'delete':
            response = requests.get(url, headers=headers)
            modified = {}
            data = response.json()
            for entry in data['rows']:
                if (entry['test'] == test_modified) and (entry['target'] == target):
                    new_entry = entry
                    new_entry['value'] = round(float(entry['value']) / 100 + float(entry['value']), 1)
                    new_entry.pop('project_id', None)
                    modified[entry['id']] = new_entry
            for entry in modified:
                params = {"id[]": str(entry)}
                delete_response = requests.delete(url, headers=headers, params=params)
                print("Deleted Data:", delete_response.status_code)
                time.sleep(3)

            return {'statusCode': 204, 'body': f"Thresholds deleted"}
        else:
            print("Valid action parameter not provided")
            return {'statusCode': 400, 'body': f"Valid action parameter not provided"}

    except Exception as e:
        print("Failed to update thresholds")
        print(e)
        return {'statusCode': 400, 'body': f"{e}"}


def handler_local(event={}, context=None):
    print("[INFO] Starting")
    try:
        action = 'delete'
        project_id = '27'
        carrier_url = 'https://platform.getcarrier.io'
        carrier_token = 'eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJ1dWlkIjoiNDZlN2ZkN2YtMTEzZC00Y2FiLTkwM2UtODFjYzJkYjMwZTNhIn0.DqF_1PMeYk-Uglf61Y4a60f7liL4zvpDFHH9-bE0_9UviEaeCvZk6WU8nj2R5TwsXlMexUftk6j0igbuXC90Uw'
        target = "response_time"
        aggregation = 'pct95'
        comparison = 'gte'
        test_modified = "VCT_STG_APIM"
        change_in_percentages = '5.0'
        env = 'demo'
        url = carrier_url + '/api/v1/backend_performance/thresholds/' + project_id
        baseline_url = carrier_url + '/api/v1/backend_performance/baseline/' + project_id + '?test_name=' + test_modified + '&env=' + env
        headers = {
            'Authorization': f'Bearer {carrier_token}',
            'Accept': '*/*'
        }

        if action == 'create':

            baseline_requests = {}
            response_baseline = requests.get(baseline_url, headers=headers)
            time.sleep(3)
            data_baseline = response_baseline.json()
            max_threshold = 0.0
            for entry in data_baseline['baseline']:
                if entry[aggregation] > max_threshold:
                    max_threshold = entry[aggregation]
                new_value = round(
                    float(entry[aggregation]) / 100.0 * float(change_in_percentages) + float(entry[aggregation]), 1)
                baseline_requests[entry['request_name']] = {'test': entry['simulation'], 'environment': entry['env'],
                                                            'scope': entry['request_name'], 'target': target,
                                                            'aggregation': aggregation, 'comparison': comparison,
                                                            'value': new_value}
                create_response = requests.post(url + '/', headers=headers,
                                                json=baseline_requests[entry['request_name']])
                print("Created response: " + str(create_response.status_code))
                time.sleep(3)
            every_value = round(float(max_threshold) / 100.0 * float(change_in_percentages) + float(max_threshold), 1)
            every = {'test': test_modified, 'environment': env, 'scope': 'every', 'target': target,
                     'aggregation': aggregation, 'comparison': comparison, 'value': every_value}
            every_response = requests.post(url, headers=headers, json=every)
            return {'statusCode': 200, 'body': f"Thresholds created"}

        elif action == 'update':
            response = requests.get(url, headers=headers)
            modified = {}
            if response.status_code == 200:
                data = response.json()
                for entry in data['rows']:
                    if (entry['test'] == test_modified) and (entry['target'] == target):
                        new_entry = entry
                        new_entry['value'] = round(
                            float(entry['value']) / float(100.0) * float(change_in_percentages) + float(entry['value']),
                            1)
                        new_entry.pop('project_id', None)
                        modified[entry['id']] = new_entry
                for entry in modified:
                    update_response = requests.put(url + '/' + str(entry), headers=headers, json=modified[entry])
                    print("Updated response:", update_response.status_code)
                    time.sleep(3)
            else:
                print('Failed to get data:', response.status_code, response.text)
            return {'statusCode': 200, 'body': f"Thresholds updated"}

        elif action == 'delete':
            response = requests.get(url, headers=headers)
            modified = {}
            data = response.json()
            for entry in data['rows']:
                if (entry['test'] == test_modified) and (entry['target'] == target):
                    new_entry = entry
                    new_entry['value'] = round(float(entry['value']) / 100 + float(entry['value']), 1)
                    new_entry.pop('project_id', None)
                    modified[entry['id']] = new_entry
            for entry in modified:
                params = {"id[]": str(entry)}
                delete_response = requests.delete(url, headers=headers, params=params)
                print("Deleted Data:", delete_response.status_code)
                time.sleep(3)

            return {'statusCode': 204, 'body': f"Thresholds deleted"}
        else:
            print("Valid action parameter not provided")
            return {'statusCode': 400, 'body': f"Valid action parameter not provided"}

    except Exception as e:
        print("Failed to update thresholds")
        print(e)
        return {'statusCode': 400, 'body': f"{e}"}


