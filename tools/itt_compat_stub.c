/*
 * Runtime compatibility shim for PyTorch builds linked against the legacy
 * Intel JIT profiling API.  This workspace has a modern MKL runtime without
 * libittnotify.so.  The three functions below are profiler-only; returning
 * "not active" disables profiling and does not change tensor execution.
 *
 * Build locally and use only through LD_PRELOAD for the affected process.
 */
unsigned int iJIT_GetNewMethodID(void) { return 0U; }
int iJIT_IsProfilingActive(void) { return 0; }
int iJIT_NotifyEvent(int event_type, void *event_data) {
    (void)event_type;
    (void)event_data;
    return 0;
}
