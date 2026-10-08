/*
 * PyTorch 2.5.1's CPU library references Intel's optional VTune JIT profiling
 * hook.  Recent MKL distributions no longer ship that symbol.  Reloc3r does
 * not use VTune, so a no-op implementation lets the model load without
 * changing the shared Conda environment.
 */
int iJIT_NotifyEvent(int event_type, void *event_data) {
    (void)event_type;
    (void)event_data;
    return 0;
}

int iJIT_IsProfilingActive(void) {
    return 0;
}

unsigned int iJIT_GetNewMethodID(void) {
    return 0;
}
