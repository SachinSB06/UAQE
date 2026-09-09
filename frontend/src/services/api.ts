/**
 * UAQE Frontend API Service.
 * Centralized HTTP requests, real multipart file & folder uploads, and Server-Sent Events client.
 */

import {
  SystemStatus,
  JobSummary,
  JobDetail,
  TelemetryResponse,
  OptimizationEvent,
  ModelUploadResult,
  DatasetUploadResult,
} from '../types/api';

const API_BASE = '/api';

export async function fetchStatus(): Promise<SystemStatus> {
  const res = await fetch(`${API_BASE}/status`);
  if (!res.ok) throw new Error(`Failed to fetch system status: ${res.statusText}`);
  return res.json();
}

export async function fetchJobs(): Promise<JobSummary[]> {
  const res = await fetch(`${API_BASE}/jobs`);
  if (!res.ok) throw new Error(`Failed to fetch jobs: ${res.statusText}`);
  return res.json();
}

export async function fetchJobDetail(jobId: string): Promise<JobDetail> {
  const res = await fetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}`);
  if (!res.ok) throw new Error(`Failed to fetch job ${jobId}: ${res.statusText}`);
  return res.json();
}

export async function fetchJobTelemetry(jobId: string): Promise<TelemetryResponse> {
  const res = await fetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}/telemetry`);
  if (!res.ok) throw new Error(`Failed to fetch telemetry for job ${jobId}: ${res.statusText}`);
  return res.json();
}

export async function fetchSamples(): Promise<{ models: any[]; datasets: any[] }> {
  const res = await fetch(`${API_BASE}/uploads/samples`);
  if (!res.ok) throw new Error(`Failed to fetch verified samples: ${res.statusText}`);
  return res.json();
}

export async function uploadModelFile(
  file: File,
  onProgress?: (progress: { loaded: number; total: number; pct: number }) => void
): Promise<ModelUploadResult> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}/uploads/model`);

    if (xhr.upload && onProgress) {
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const pct = Math.round((event.loaded / event.total) * 100);
          onProgress({ loaded: event.loaded, total: event.total, pct });
        }
      };
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText);
          resolve(data);
        } catch (e) {
          reject(new Error('Invalid JSON response from model upload'));
        }
      } else {
        try {
          const err = JSON.parse(xhr.responseText);
          reject(new Error(err.detail || `Model upload failed (${xhr.status})`));
        } catch {
          reject(new Error(`Model upload failed with HTTP ${xhr.status}: ${xhr.statusText}`));
        }
      }
    };

    xhr.onerror = () => reject(new Error('Network error during model upload'));

    const formData = new FormData();
    formData.append('file', file);
    xhr.send(formData);
  });
}

export async function uploadDatasetFolder(
  files: File[],
  relativePaths: string[],
  folderName: string = 'dataset',
  onProgress?: (progress: { loadedFiles: number; totalFiles: number; loadedBytes: number; totalBytes: number; pct: number }) => void
): Promise<DatasetUploadResult> {
  return new Promise((resolve, reject) => {
    let totalBytes = 0;
    for (const f of files) totalBytes += f.size;
    const totalMB = (totalBytes / (1024 * 1024)).toFixed(2);

    // Diagnostics before upload (Requirement 11)
    console.log('[UAQE] Dataset upload starting', {
      fileCount: files.length,
      totalBytes,
      totalMB,
      folderName,
      firstPaths: relativePaths.slice(0, 10),
    });

    const xhr = new XMLHttpRequest();
    // 10 minutes timeout (Requirement 10.B)
    xhr.timeout = 600000;
    xhr.open('POST', `${API_BASE}/uploads/dataset`);

    // DO NOT set Content-Type header manually - let browser generate boundary (Requirement 10.C)

    if (xhr.upload && onProgress) {
      let lastLoggedPct = -1;
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const pct = Math.round((event.loaded / event.total) * 100);
          const loadedFilesEst = Math.min(files.length, Math.round((event.loaded / (event.total || 1)) * files.length));
          onProgress({
            loadedFiles: loadedFilesEst,
            totalFiles: files.length,
            loadedBytes: event.loaded,
            totalBytes: event.total,
            pct,
          });

          // Log periodically during upload (Requirement 11)
          if (pct !== lastLoggedPct && pct % 10 === 0) {
            lastLoggedPct = pct;
            console.log(`[UAQE] Dataset upload progress: ${pct}% (${(event.loaded / (1024 * 1024)).toFixed(2)} MB / ${(event.total / (1024 * 1024)).toFixed(2)} MB)`);
          }
        }
      };
    }

    xhr.onload = () => {
      console.log(`[UAQE] Dataset upload complete. HTTP Status: ${xhr.status} ${xhr.statusText}`);
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText);
          resolve(data);
        } catch {
          reject(new Error('Invalid JSON response received from dataset folder upload.'));
        }
      } else {
        // Detailed error classification (Requirement 12)
        let errorDetail = '';
        try {
          const parsed = JSON.parse(xhr.responseText);
          errorDetail = parsed.detail || (parsed.error && parsed.error.message) || parsed.message || '';
          if (parsed.error && parsed.error.details) {
            errorDetail += ` (${parsed.error.details})`;
          }
        } catch {
          errorDetail = xhr.responseText ? xhr.responseText.substring(0, 300) : xhr.statusText;
        }

        if (xhr.status === 400) {
          reject(new Error(`Bad Request (HTTP 400): ${errorDetail || 'Invalid dataset format or relative paths'}`));
        } else if (xhr.status === 413) {
          reject(new Error(`Payload Too Large (HTTP 413): ${errorDetail || 'Upload exceeds server limit'}`));
        } else if (xhr.status === 500) {
          reject(new Error(`Server Internal Error (HTTP 500): ${errorDetail || 'Dataset processing failed on backend'}`));
        } else if (xhr.status === 502 || xhr.status === 504) {
          reject(new Error(`Gateway / Proxy Error (HTTP ${xhr.status}): Backend server unreachable or timed out.`));
        } else {
          reject(new Error(`Upload failed with HTTP ${xhr.status}: ${errorDetail || xhr.statusText}`));
        }
      }
    };

    xhr.ontimeout = () => {
      console.error('[UAQE] Dataset upload timed out after 600 seconds');
      reject(new Error('Upload timeout: Dataset upload exceeded the 10-minute timeout limit.'));
    };

    xhr.onabort = () => {
      console.warn('[UAQE] Dataset upload aborted by user/client');
      reject(new Error('Upload aborted: The dataset upload was cancelled.'));
    };

    xhr.onerror = () => {
      console.error('[UAQE] Dataset upload XHR network error occurred');
      if (typeof window !== 'undefined' && window.navigator && !window.navigator.onLine) {
        reject(new Error('Network offline: Client lost internet/network connectivity.'));
      } else {
        reject(new Error('Connection/Network reset: The backend server dropped the connection or a CORS/proxy error occurred.'));
      }
    };

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      formData.append('files', f, f.name);
    }
    formData.append('relative_paths', JSON.stringify(relativePaths));
    formData.append('folder_name', folderName);

    xhr.send(formData);
  });
}

export async function uploadDatasetZip(
  file: File,
  onProgress?: (progress: { loaded: number; total: number; pct: number }) => void
): Promise<DatasetUploadResult> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}/uploads/dataset/zip`);

    if (xhr.upload && onProgress) {
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const pct = Math.round((event.loaded / event.total) * 100);
          onProgress({ loaded: event.loaded, total: event.total, pct });
        }
      };
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText);
          resolve(data);
        } catch (e) {
          reject(new Error('Invalid JSON response from dataset zip upload'));
        }
      } else {
        try {
          const err = JSON.parse(xhr.responseText);
          reject(new Error(err.detail || `Dataset zip upload failed (${xhr.status})`));
        } catch {
          reject(new Error(`Dataset zip upload failed with HTTP ${xhr.status}: ${xhr.statusText}`));
        }
      }
    };

    xhr.onerror = () => reject(new Error('Network error during dataset zip upload'));

    const formData = new FormData();
    formData.append('file', file);
    xhr.send(formData);
  });
}

export async function analyzeModelAndDataset(params: {
  model_upload_id?: string;
  model_id?: string;
  model_path?: string;
  dataset_upload_id?: string;
  dataset_id?: string;
  dataset_path?: string;
  target?: string;
  profile?: string;
  calib_samples?: number;
  test_samples?: number;
}): Promise<any> {
  const res = await fetch(`${API_BASE}/jobs/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Analysis failed: ${res.statusText}`);
  }
  return res.json();
}

export async function planOptimization(params: {
  model_upload_id?: string;
  model_id?: string;
  model_path?: string;
  dataset_upload_id?: string;
  dataset_id?: string;
  dataset_path?: string;
  target?: string;
  profile?: string;
  max_candidates?: number;
  calib_samples?: number;
  test_samples?: number;
}): Promise<any> {
  const res = await fetch(`${API_BASE}/jobs/plan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Planning failed: ${res.statusText}`);
  }
  return res.json();
}

export async function startOptimization(params: {
  model_upload_id?: string;
  model_id?: string;
  model_path?: string;
  dataset_upload_id?: string;
  dataset_id?: string;
  dataset_path?: string;
  target?: string;
  profile?: string;
  max_candidates?: number;
  max_budget?: number;
  auto_approve?: boolean;
  calib_samples?: number;
  test_samples?: number;
}): Promise<{ job_id: string; status: string; message: string; events_url: string }> {
  const res = await fetch(`${API_BASE}/jobs/optimize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Optimization launch failed: ${res.statusText}`);
  }
  return res.json();
}

export function getArtifactDownloadUrl(jobId: string, filename: string, candidateId?: string): string {
  const base = `${API_BASE}/jobs/${encodeURIComponent(jobId)}/download/${encodeURIComponent(filename)}`;
  return candidateId ? `${base}?candidate_id=${encodeURIComponent(candidateId)}` : base;
}

export function getCandidateDownloadUrl(jobId: string, candidateId: string): string {
  return `${API_BASE}/jobs/${encodeURIComponent(jobId)}/candidates/${encodeURIComponent(candidateId)}/download`;
}

export function subscribeToJobEvents(
  jobId: string,
  onEvent: (event: OptimizationEvent) => void,
  onError?: (err: any) => void
): () => void {
  let isClosed = false;
  const eventSource = new EventSource(`${API_BASE}/jobs/${encodeURIComponent(jobId)}/events`);

  const closeStream = () => {
    if (!isClosed) {
      isClosed = true;
      eventSource.close();
    }
  };

  eventSource.onmessage = (e) => {
    try {
      const data: OptimizationEvent = JSON.parse(e.data);
      onEvent(data);

      const typeUpper = (data.type || '').toUpperCase();
      if (
        typeUpper === 'COMPLETE' ||
        typeUpper === 'OPTIMIZATION_COMPLETED' ||
        typeUpper === 'FINAL_RESULT' ||
        typeUpper === 'ERROR' ||
        typeUpper === 'FAILED'
      ) {
        closeStream();
      }
    } catch (err) {
      console.error('Failed to parse SSE event:', err);
    }
  };

  eventSource.onerror = (err) => {
    if (isClosed) return;
    if (onError) onError(err);
  };

  return closeStream;
}

export async function compareJobs(jobA: string, jobB: string): Promise<any> {
  const res = await fetch(`${API_BASE}/jobs/compare?job_a=${encodeURIComponent(jobA)}&job_b=${encodeURIComponent(jobB)}`);
  if (!res.ok) throw new Error(`Failed to compare jobs: ${res.statusText}`);
  return res.json();
}

export const apiService = {
  getSystemStatus: fetchStatus,
  getJobs: fetchJobs,
  getJobDetail: fetchJobDetail,
  getTelemetry: fetchJobTelemetry,
  getSamples: fetchSamples,
  uploadModelFile,
  uploadDatasetFolder,
  uploadDatasetZip,
  analyzeModel: analyzeModelAndDataset,
  planOptimization,
  launchOptimization: startOptimization,
  getArtifactDownloadUrl,
  getCandidateDownloadUrl,
  streamJobEvents: subscribeToJobEvents,
  compareJobs,
};

export default apiService;
