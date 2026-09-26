/** Request sequencing for scene loads: one live request, superseded ones aborted, late responses ignored. */

/**
 * `fetchScene(payload, signal)` performs the request and returns the validated
 * scene. `onStart` runs before the request, `onSuccess(scene, payload)` for the
 * newest request only, `onError(error)` for a failure of the newest request
 * (never for an abort), and `onSettled()` once the newest request has ended
 * either way. `schedule(run, delay)` debounces a later load; `cancelScheduled`
 * drops it, and every `load` drops it too.
 */
export function createSceneLoader({ fetchScene, onStart, onSuccess, onError, onSettled }) {
  let controller = null;
  let sequence = 0;
  let timer = null;

  function cancelScheduled() {
    window.clearTimeout(timer);
    timer = null;
  }

  function schedule(run, delay = 260) {
    cancelScheduled();
    timer = window.setTimeout(() => {
      timer = null;
      run();
    }, delay);
  }

  async function load(payload) {
    cancelScheduled();
    controller?.abort();
    const current = new AbortController();
    const ticket = ++sequence;
    controller = current;
    onStart(payload);
    try {
      const scene = await fetchScene(payload, current.signal);
      if (ticket !== sequence) return;
      onSuccess(scene, payload);
    } catch (error) {
      if (error.name !== "AbortError" && ticket === sequence) onError(error);
    } finally {
      if (ticket === sequence) {
        controller = null;
        onSettled();
      }
    }
  }

  return { load, schedule, cancelScheduled };
}
