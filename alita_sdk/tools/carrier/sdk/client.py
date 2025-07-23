import json
import logging
import os
import requests
from typing import Any, Dict, List, Tuple
import shutil

from pydantic import BaseModel, Field, model_validator

from .data_models import CarrierCredentials
from .endpoints import _EndpointManager
from .exceptions import CarrierAPIError
from ..utils.utils import get_latest_log_file

logger = logging.getLogger("carrier_sdk.client")


class CarrierClient(BaseModel):
    """
    The primary client for interacting with the Carrier API. It handles
    session management, authentication, and orchestrates API calls.
    """
    credentials: CarrierCredentials
    session: requests.Session = Field(default_factory=requests.Session, exclude=True)
    endpoints: _EndpointManager = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Enhanced request method with comprehensive error handling."""
        full_url = f"{self.credentials.url.rstrip('/')}/{endpoint.lstrip('/')}"

        # Add request ID for tracking
        REQUEST_ID_MODULO = 10000
        request_id = f"{method.upper()}_{hash(full_url) % REQUEST_ID_MODULO}"
        logger.debug(f"[{request_id}] Making API request: {method.upper()} {full_url}")

        try:
            # Handle file uploads
            headers = self.session.headers.copy()
            if 'files' in kwargs or 'data' in kwargs:
                headers.pop('Content-Type', None)

            response = self.session.request(method, full_url, headers=headers, **kwargs)

            # Log response details
            logger.debug(f"[{request_id}] Response status: {response.status_code}")

            response.raise_for_status()
            return response

        except requests.HTTPError as http_err:
            # Enhanced error logging with request context
            error_context = {
                'request_id': request_id,
                'method': method.upper(),
                'url': full_url,
                'status_code': http_err.response.status_code,
                'response_snippet': http_err.response.text[:500]
            }
            logger.error(f"HTTP Error: {error_context}")

            raise CarrierAPIError(
                f"API request failed [{request_id}]",
                status_code=http_err.response.status_code,
                response_text=http_err.response.text
            ) from http_err

        except requests.RequestException as req_err:
            logger.error(f"[{request_id}] Network error: {req_err}")
            raise CarrierAPIError(f"Network error occurred [{request_id}]: {req_err}") from req_err

    @model_validator(mode='after')
    def initialize_client(self, __context: Any) -> "CarrierClient":
        """Initializes the session headers and the endpoint manager after Pydantic validation."""
        headers = {
            'Authorization': f'Bearer {self.credentials.token}',
            'Content-Type': 'application/json',
        }
        self.session.headers.update(headers)
        self.endpoints = _EndpointManager(project_id=self.credentials.project_id)
        logger.info("CarrierClient initialized with EndpointManager and authenticated session.")
        return self

    # ADD TO CarrierClient class in paste.txt

    def create_ticket(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Creates a new ticket in the system."""
        endpoint = self.endpoints.build_endpoint('create_ticket')
        response = self._json_request('post', endpoint, json=ticket_data)

        if not response or "item" not in response:
            logger.warning(f"Unexpected ticket creation response: {response}")
            raise CarrierAPIError("Carrier did not return a valid ticket response")

        return response

    def fetch_tickets(self, board_id: str) -> List[Dict[str, Any]]:
        """Fetches tickets from a specific board."""
        endpoint = self.endpoints.build_endpoint('list_tickets', board_id=board_id)
        return self._json_request('get', endpoint).get("rows", [])

    def get_reports_list(self) -> List[Dict[str, Any]]:
        """Gets list of performance reports."""
        endpoint = self.endpoints.build_endpoint('list_reports')
        return self._json_request('get', endpoint).get("rows", [])

    def get_tests_list(self) -> List[Dict[str, Any]]:
        """Gets list of performance tests."""
        endpoint = self.endpoints.build_endpoint('list_tests')
        return self._json_request('get', endpoint).get("rows", [])

    def create_test(self, test_data: Dict[str, Any]) -> requests.Response:
        """Creates a new performance test using form-data."""
        endpoint = self.endpoints.build_endpoint('create_test')

        # Use form-data encoding as required by the API
        form_data = {"data": json.dumps(test_data)}

        # Remove Content-Type header temporarily for multipart
        original_headers = self.session.headers.copy()
        if 'Content-Type' in self.session.headers:
            del self.session.headers['Content-Type']

        try:
            response = self._request('post', endpoint, data=form_data)
            return response
        finally:
            # Restore original headers
            self.session.headers.update(original_headers)

    def run_test(self, test_id: str, json_body: Dict[str, Any]) -> str:
        """Runs a performance test and returns result ID."""
        endpoint = self.endpoints.build_endpoint('run_test', test_id=test_id)
        response = self._json_request('post', endpoint, json=json_body)
        return response.get("result_id", "")

    def get_integrations(self, name: str) -> Dict[str, Any]:
        """Gets integration details by name."""
        endpoint = self.endpoints.build_endpoint('list_integrations', name=name)
        return self._json_request('get', endpoint)

    def get_available_locations(self) -> Dict[str, Any]:
        """Gets available test locations."""
        endpoint = self.endpoints.build_endpoint('get_available_locations')
        return self._json_request('get', endpoint)

    def get_engagements_list(self) -> List[Dict[str, Any]]:
        """Gets list of engagements."""
        endpoint = self.endpoints.build_endpoint('list_engagements')
        return self._json_request('get', endpoint).get("items", [])

    def _json_request(self, method: str, endpoint: str, **kwargs) -> Any:
        """PRIVATE method for requests expecting a JSON response."""
        response = self._request(method, endpoint, **kwargs)
        try:
            return response.json()
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to decode JSON from response. URL: {response.url}, Response Text: {response.text}")
            raise CarrierAPIError(
                "Server returned a non-JSON response",
                status_code=response.status_code,
                response_text=response.text
            ) from json_err

    # In your existing CarrierClient class

    def get_report_info(self, report_id: str) -> Dict[str, Any]:
        """
        🔍 Get report information by ID with comprehensive logging.

        FIXED: Use existing _json_request method instead of returning raw Response.
        """
        logger.info(f"🔍 Getting report info for ID: {report_id}")

        try:
            # Build endpoint for reports list
            endpoint = self.endpoints.build_endpoint('list_reports')
            logger.debug(f"   Built endpoint: {endpoint}")

            # Add query parameter for specific report
            params = {'report_id': report_id}
            logger.debug(f"   Query params: {params}")

            # ✅ FIX: Use existing _json_request method instead of _request
            response_data = self._json_request('GET', endpoint, params=params)
            logger.debug(f"   Response type: {type(response_data)}")
            logger.debug(
                f"   Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Not a dict'}")

            # Validate response structure
            if not isinstance(response_data, dict):
                logger.error(f"❌ Expected dict response, got {type(response_data)}")
                logger.error(f"   Response content: {response_data}")
                raise CarrierAPIError(f"Invalid response format: expected dict, got {type(response_data)}")

            # Check if this is a single report or list response
            if 'name' in response_data and 'id' in response_data:
                # Direct report info
                logger.info(f"✅ Got direct report info: {response_data.get('name')}")
                return response_data

            elif 'rows' in response_data and isinstance(response_data['rows'], list):
                # List response - find the specific report
                logger.info(f"✅ Got list response with {len(response_data['rows'])} reports")

                for report in response_data['rows']:
                    if str(report.get('id')) == str(report_id):
                        logger.info(f"✅ Found matching report: {report.get('name', 'Unknown')}")
                        return report

                # Report not found in list
                logger.error(f"❌ Report {report_id} not found in {len(response_data['rows'])} reports")
                available_ids = [str(r.get('id', 'None')) for r in response_data['rows'][:5]]
                logger.error(f"   Available report IDs (first 5): {available_ids}")
                raise CarrierAPIError(f"Report {report_id} not found in available reports")

            else:
                logger.error(f"❌ Unexpected response structure")
                logger.error(f"   Response keys: {list(response_data.keys())}")
                logger.error(f"   Response sample: {str(response_data)[:200]}")
                raise CarrierAPIError(f"Unexpected response structure from API")

        except Exception as e:
            logger.error(f"❌ Failed to get report info for {report_id}: {str(e)}")
            raise

    def get_report_file_name(self, report_id: str, extract_to: str = "/tmp") -> Tuple[Dict[str, Any], str, str]:
        """
        📄 Get report file information and download logs with comprehensive logging.

        FIXED: Proper error handling and response validation.
        """
        logger.info(f"📄 Getting report file name for ID: {report_id}")
        logger.info(f"   Extract to: {extract_to}")

        try:
            # Step 1: Get report info with validation
            logger.info("📋 Step 1: Getting report information...")
            report_info = self.get_report_info(report_id)

            # Validate report_info structure
            if not isinstance(report_info, dict):
                logger.error(f"❌ Expected dict from get_report_info, got {type(report_info)}")
                raise CarrierAPIError(f"Invalid report info format: {type(report_info)}")

            # Check required fields
            required_fields = ['name', 'build_id']
            missing_fields = [field for field in required_fields if field not in report_info]
            if missing_fields:
                logger.error(f"❌ Missing required fields in report info: {missing_fields}")
                logger.error(f"   Available fields: {list(report_info.keys())}")
                raise CarrierAPIError(f"Report info missing required fields: {missing_fields}")

            logger.info(f"✅ Got report info: {report_info.get('name')}")

            # Step 2: Extract bucket and file information
            logger.info("🗃️ Step 2: Processing bucket information...")
            bucket_name = report_info["name"].replace("_", "").replace(" ", "").lower()
            report_archive_prefix = f"reports_test_results_{report_info['build_id']}"
            lg_type = report_info.get("lg_type", "gatling")  # Default to gatling

            logger.info(f"   Bucket name: {bucket_name}")
            logger.info(f"   Archive prefix: {report_archive_prefix}")
            logger.info(f"   Load generator type: {lg_type}")

            # Step 3: Get files from bucket
            logger.info("📁 Step 3: Getting files from bucket...")
            try:
                bucket_endpoint = self.endpoints.build_endpoint('list_artifacts', bucket_name=bucket_name)
                files_info = self._json_request('GET', bucket_endpoint)

                if 'rows' not in files_info:
                    logger.error(f"❌ No 'rows' field in bucket response")
                    logger.error(f"   Response keys: {list(files_info.keys())}")
                    raise CarrierAPIError(f"Invalid bucket response format")

                logger.info(f"✅ Found {len(files_info['rows'])} files in bucket")

            except Exception as e:
                logger.error(f"❌ Failed to get bucket files: {str(e)}")
                raise CarrierAPIError(f"Failed to access bucket {bucket_name}: {str(e)}")

                # Step 4: Filter relevant report files
            logger.info("🔍 Step 4: Filtering report files...")
            file_list = [file_data["name"] for file_data in files_info["rows"]]
            report_files_list = []

            for file_name in file_list:
                if file_name.startswith(report_archive_prefix) and "excel_report" not in file_name:
                    report_files_list.append(file_name)
                    logger.debug(f"   ✅ Found report file: {file_name}")

            if not report_files_list:
                logger.error(f"❌ No report files found with prefix: {report_archive_prefix}")
                logger.error(f"   Available files: {file_list[:10]}")  # Show first 10 files
                raise CarrierAPIError(f"No report files found for report {report_id}")

            logger.info(f"✅ Found {len(report_files_list)} report files to process")

            # Step 5: Download and merge reports
            logger.info("⬇️ Step 5: Downloading and merging reports...")
            try:
                test_log_file_path, errors_log_file_path = self.download_and_merge_reports(
                    report_files_list, lg_type, bucket_name, extract_to
                )

                # Validate that files were created
                if not os.path.exists(test_log_file_path):
                    logger.error(f"❌ Test log file was not created: {test_log_file_path}")
                    raise CarrierAPIError(f"Failed to create test log file")

                logger.info(f"✅ Test log file created: {test_log_file_path}")
                logger.info(f"✅ Error log file: {errors_log_file_path}")

            except Exception as e:
                logger.error(f"❌ Failed to download and merge reports: {str(e)}")
                raise CarrierAPIError(f"Failed to process report files: {str(e)}")

            logger.info(f"🎉 Successfully processed report {report_id}")
            return report_info, test_log_file_path, errors_log_file_path

        except CarrierAPIError:
            # Re-raise CarrierAPIError as-is
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error in get_report_file_name: {str(e)}")
            logger.error(f"   Report ID: {report_id}")
            logger.error(f"   Extract path: {extract_to}")
            raise CarrierAPIError(f"Unexpected error processing report {report_id}: {str(e)}")

    def download_artifact_to_file(self, bucket: str, file_name: str, extract_to: str = "/tmp") -> str:
        """Downloads a single artifact from a bucket and saves it to a local file."""
        endpoint = self.endpoints.build_endpoint('download_artifact', bucket_name=bucket, file_name=file_name)
        response = self._request('get', endpoint)
        local_file_path = os.path.join(extract_to, file_name)
        try:
            with open(local_file_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded artifact '{file_name}' to '{local_file_path}' ({len(response.content)} bytes).")
            return local_file_path
        except IOError as io_err:
            logger.error(f"Failed to write downloaded artifact to disk at {local_file_path}: {io_err}")
            raise CarrierAPIError(f"Failed to save artifact file: {io_err}") from io_err

    def upload_artifact(self, bucket_name: str, file_path: str) -> bool:
        """Uploads a single local file to a specified bucket."""
        endpoint = self.endpoints.build_endpoint('upload_artifact', bucket_name=bucket_name)
        file_name_on_server = os.path.basename(file_path)
        original_headers = self.session.headers.copy()
        if 'Content-Type' in self.session.headers:
            del self.session.headers['Content-Type']

        try:
            with open(file_path, 'rb') as f:
                files = {'file': (file_name_on_server, f)}
                # Use _request directly as the response may not be JSON.
                self._request('post', endpoint, files=files)
            logger.info(f"Successfully uploaded file '{file_name_on_server}' to bucket '{bucket_name}'.")
            return True
        except FileNotFoundError:
            logger.error(f"Upload failed: local file not found at {file_path}")
            raise CarrierAPIError(f"Cannot upload file, not found at: {file_path}")
        except CarrierAPIError:
            # Error is already logged by _request, so just log the context and return False.
            logger.error(f"Upload of '{file_name_on_server}' to bucket '{bucket_name}' failed at the API level.")
            return False
        finally:
            # Restore original headers
            self.session.headers.update(original_headers)

    def add_tag_to_report(self, report_id: str, tag_name: str) -> requests.Response:
        """Restored legacy method to add tags to reports."""
        endpoint = self.endpoints.build_endpoint('add_tag_to_report', report_id=report_id)
        data = {"tags": [{"title": tag_name, "hex": "#5933c6"}]}
        # Use _request as the response might not always be JSON.
        return self._request('post', endpoint, json=data)

    # ADD TO CarrierClient class in paste.txt

    def download_and_unzip_reports(self, file_name: str, bucket: str, extract_to: str = "/tmp") -> str:
        """Downloads and extracts report archives."""
        import zipfile
        import shutil

        # Use existing download method
        local_file_path = self.download_artifact_to_file(bucket, file_name, extract_to)

        extract_dir = local_file_path.replace('.zip', '')

        # Clean up existing extraction directory
        try:
            shutil.rmtree(extract_dir)
        except Exception as e:
            logger.debug(f"No existing extract directory to clean: {e}")

        # Extract the zip file
        with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Clean up the zip file
        if os.path.exists(local_file_path):
            os.remove(local_file_path)

        logger.info(f"Extracted report archive to: {extract_dir}")
        return extract_dir

    def download_and_merge_reports(self, report_files_list: List[str], lg_type: str,
                                   bucket: str, extract_to: str = "/tmp") -> tuple:
        """Downloads multiple report files and merges them."""
        # Generate merged file names
        if lg_type == "jmeter":
            summary_log_file_path = f"summary_{bucket}_jmeter.jtl"
            error_log_file_path = f"error_{bucket}_jmeter.log"
        else:
            summary_log_file_path = f"summary_{bucket}_simulation.log"
            error_log_file_path = f"error_{bucket}_simulation.log"

        extracted_reports = []

        # Download and extract each report file
        for file_name in report_files_list:
            try:
                extract_dir = self.download_and_unzip_reports(file_name, bucket, extract_to)
                extracted_reports.append(extract_dir)
            except Exception as e:
                logger.error(f"Failed to download/extract {file_name}: {e}")
                continue

        # Merge log files
        self._merge_log_files(summary_log_file_path, extracted_reports, lg_type)

        # Merge error files
        try:
            self._merge_error_files(error_log_file_path, extracted_reports)
        except Exception as e:
            logger.error(f"Failed to merge error logs: {e}")

        # Clean up extracted directories
        for extract_dir in extracted_reports:
            try:
                shutil.rmtree(extract_dir)
            except Exception as e:
                logger.error(f"Failed to clean up {extract_dir}: {e}")

        return summary_log_file_path, error_log_file_path

    def _merge_log_files(self, summary_file: str, extracted_reports: List[str], lg_type: str):
        """Merges multiple log files into a single summary file."""
        with open(summary_file, mode='w') as summary:
            for i, log_dir in enumerate(extracted_reports):
                if lg_type == "jmeter":
                    report_file = f"{log_dir}/jmeter.jtl"
                else:
                    report_file = get_latest_log_file(log_dir, "simulation.log")

                try:
                    with open(report_file, mode='r') as f:
                        lines = f.readlines()
                        if i == 0:
                            # Write all lines from first file (including header)
                            summary.writelines(lines)
                        else:
                            # Skip header for subsequent files
                            summary.writelines(lines[1:])
                except Exception as e:
                    logger.error(f"Failed to read log file {report_file}: {e}")

    def _merge_error_files(self, error_file: str, extracted_reports: List[str]):
        """Merges error log files from multiple extractions."""
        with open(error_file, mode='w') as summary_errors:
            for log_dir in extracted_reports:
                error_report_file = f"{log_dir}/simulation-errors.log"
                try:
                    with open(error_report_file, mode='r') as f:
                        lines = f.readlines()
                        summary_errors.writelines(lines)
                except Exception as e:
                    logger.debug(f"No error file found at {error_report_file}: {e}")

    def get_report_file_log(self, bucket: str, file_name: str) -> str:
        """Downloads a specific report file from bucket."""
        endpoint = self.endpoints.build_endpoint('download_artifact_default',
                                                 bucket_name=bucket, file_name=file_name)

        # Add S3 config parameters
        params = {'integration_id': 1, 'is_local': False}
        response = self._request('get', endpoint, params=params)

        file_path = f"/tmp/{file_name}"
        try:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded report file to: {file_path}")
            return file_path
        except IOError as io_err:
            logger.error(f"Failed to write report file to {file_path}: {io_err}")
            raise CarrierAPIError(f"Failed to save report file: {io_err}") from io_err

    def upload_file(self, bucket_name: str, file_path: str) -> bool:
        """Uploads a file to the specified bucket with S3 configuration."""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise CarrierAPIError(f"File not found: {file_path}")
        endpoint = self.endpoints.build_endpoint('upload_artifact', bucket_name=bucket_name)
        params = {'integration_id': 1, 'is_local': False}

        # Temporarily remove Content-Type header for multipart
        original_headers = self.session.headers.copy()
        if 'Content-Type' in self.session.headers:
            del self.session.headers['Content-Type']

        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f)}
                st = self._request('post', endpoint, params=params, files=files)
            print(st.text)
            logger.info(f"Successfully uploaded {file_path} to bucket {bucket_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {file_path}: {e}")
            return False
        finally:
            # Restore original headers
            self.session.headers.update(original_headers)

    def get_ui_tests_list(self) -> List[Dict[str, Any]]:
        """Gets list of UI tests."""
        endpoint = self.endpoints.build_endpoint('list_ui_tests')
        return self._json_request('get', endpoint).get("rows", [])

    def get_ui_reports_list(self) -> List[Dict[str, Any]]:
        """Gets list of UI test reports."""
        endpoint = self.endpoints.build_endpoint('list_ui_reports')
        return self._json_request('get', endpoint).get("rows", [])

    def create_ui_test(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Creates a new UI test with form-data encoding."""
        endpoint = self.endpoints.build_endpoint('create_ui_test')

        # Use form-data for UI test creation
        form_data = {'data': json.dumps(test_data)}

        # Temporarily remove Content-Type header for multipart
        original_headers = self.session.headers.copy()
        if 'Content-Type' in self.session.headers:
            del self.session.headers['Content-Type']

        try:
            response = self._request('post', endpoint, data=form_data)
            return response.json()
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to decode JSON from UI test creation response")
            raise CarrierAPIError("Server returned non-JSON response for UI test creation") from json_err
        finally:
            # Restore original headers
            self.session.headers.update(original_headers)

    def get_ui_test_details(self, test_id: str) -> Dict[str, Any]:
        """Gets detailed UI test configuration by test ID."""
        endpoint = self.endpoints.build_endpoint('ui_test_details', test_id=test_id)
        return self._json_request('get', endpoint)

    def update_ui_test(self, test_id: str, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Updates UI test configuration and schedule."""
        endpoint = self.endpoints.build_endpoint('update_ui_test', test_id=test_id)
        return self._json_request('put', endpoint, json=test_data)

    def run_ui_test(self, test_id: str, json_body: Dict[str, Any]) -> str:
        """Runs a UI test with the given test ID and JSON body."""
        endpoint = self.endpoints.build_endpoint('run_ui_test', test_id=test_id)
        response = self._json_request('post', endpoint, json=json_body)
        return response.get("result_id", "")

    def cancel_ui_test(self, test_id: str) -> Dict[str, Any]:
        """Cancels a UI test by setting its status to Canceled."""
        endpoint = self.endpoints.build_endpoint('cancel_ui_test', test_id=test_id)

        cancel_body = {
            "test_status": {
                "status": "Canceled",
                "percentage": 100,
                "description": "Test was canceled"
            }
        }

        return self._json_request('put', endpoint, json=cancel_body)

    def get_locations(self) -> Dict[str, Any]:
        """Gets list of available locations/cloud settings."""
        endpoint = self.endpoints.build_endpoint('list_locations')
        return self._json_request('get', endpoint)
