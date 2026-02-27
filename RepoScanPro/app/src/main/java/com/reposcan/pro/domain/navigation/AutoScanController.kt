package com.reposcan.pro.domain.navigation

import com.reposcan.pro.data.location.LocationProvider
import com.reposcan.pro.data.model.RouteResponse
import com.reposcan.pro.data.remote.ApiService
import com.reposcan.pro.util.GeoUtils
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class AutoScanController @Inject constructor(
    private val arrivalDetector: ArrivalDetector,
    private val locationProvider: LocationProvider,
    private val apiService: ApiService
) {
    companion object {
        const val SCAN_DURATION_MS = 30_000L      // Scan for 30s on arrival
        const val WARMUP_RETRY_DELAY_MS = 2_000L  // Base retry delay
        const val MAX_WARMUP_RETRIES = 3
        const val SCAN_COOLDOWN_MS = 60_000L       // 1 min between auto-scans
    }

    data class NavigationState(
        val activeRoute: RouteResponse? = null,
        val currentStopIndex: Int = 0,
        val autoScanEnabled: Boolean = false,
        val isScanning: Boolean = false,
        val scanProgress: String = "",  // arrived, warming_up, scanning, complete, failed, route_complete
        val distanceToTargetM: Float = Float.MAX_VALUE,
        val isApproaching: Boolean = false
    )

    private val _state = MutableStateFlow(NavigationState())
    val state: StateFlow<NavigationState> = _state.asStateFlow()

    private var arrivalListenerJob: Job? = null
    private var distanceMonitorJob: Job? = null
    private var scanJob: Job? = null
    private val scanCooldowns = mutableMapOf<String, Long>()

    fun loadRoute(route: RouteResponse) {
        _state.update {
            NavigationState(
                activeRoute = route,
                currentStopIndex = route.currentStopIndex,
                autoScanEnabled = it.autoScanEnabled
            )
        }
    }

    fun setAutoScanEnabled(enabled: Boolean) {
        _state.update { it.copy(autoScanEnabled = enabled) }
    }

    fun startNavigation(scope: CoroutineScope) {
        val route = _state.value.activeRoute ?: return

        // Start GPS
        locationProvider.startUpdates(highAccuracy = false)

        // Start arrival geofence monitoring
        arrivalDetector.startMonitoring(
            stops = route.stops,
            currentIndex = _state.value.currentStopIndex,
            scope = scope
        )

        // Listen for arrival events
        arrivalListenerJob?.cancel()
        arrivalListenerJob = scope.launch {
            arrivalDetector.arrivalEvents.collect { event ->
                handleArrival(event, scope)
            }
        }

        // Monitor distance to current target for UI + accuracy switching
        distanceMonitorJob?.cancel()
        distanceMonitorJob = scope.launch {
            locationProvider.location.collect { loc ->
                if (!loc.isActive) return@collect
                val stop = _state.value.activeRoute
                    ?.stops
                    ?.getOrNull(_state.value.currentStopIndex)
                    ?: return@collect

                val dist = distanceBetween(
                    loc.latitude, loc.longitude,
                    stop.latitude, stop.longitude
                )
                val approaching = dist < stop.geofenceRadiusM * 3

                // Switch to high accuracy when approaching target
                if (approaching && !_state.value.isApproaching) {
                    locationProvider.switchToHighAccuracy()
                } else if (!approaching && _state.value.isApproaching) {
                    locationProvider.switchToBalancedPower()
                }

                _state.update {
                    it.copy(distanceToTargetM = dist, isApproaching = approaching)
                }
            }
        }
    }

    fun stopNavigation() {
        arrivalListenerJob?.cancel()
        distanceMonitorJob?.cancel()
        scanJob?.cancel()
        arrivalDetector.stopMonitoring()
        locationProvider.stopUpdates()
        _state.update { NavigationState() }
    }

    private suspend fun handleArrival(
        event: ArrivalDetector.ArrivalEvent,
        scope: CoroutineScope
    ) {
        // Check scan cooldown
        val lastScan = scanCooldowns[event.stopId]
        if (lastScan != null && System.currentTimeMillis() - lastScan < SCAN_COOLDOWN_MS) return

        // Notify backend of arrival
        try {
            val routeId = _state.value.activeRoute?.id ?: return
            apiService.markArrival(routeId, event.stopIndex)
        } catch (_: Exception) {
            // Continue even if backend is unreachable — arrival is local truth
        }

        _state.update {
            it.copy(currentStopIndex = event.stopIndex, scanProgress = "arrived")
        }

        // Trigger auto-scan if enabled
        if (_state.value.autoScanEnabled) {
            scanJob?.cancel()
            scanJob = scope.launch {
                performAutoScan(event)
            }
        }
    }

    private suspend fun performAutoScan(event: ArrivalDetector.ArrivalEvent) {
        val routeId = _state.value.activeRoute?.id ?: return

        // Step 1: Warmup — verify pipeline is responsive
        _state.update { it.copy(isScanning = true, scanProgress = "warming_up") }

        var ready = false
        for (attempt in 1..MAX_WARMUP_RETRIES) {
            try {
                val resp = apiService.getSystemStats()
                if (resp.isSuccessful) {
                    ready = true
                    break
                }
            } catch (_: Exception) { /* retry */ }
            if (attempt < MAX_WARMUP_RETRIES) {
                delay(WARMUP_RETRY_DELAY_MS * attempt)
            }
        }

        if (!ready) {
            _state.update { it.copy(scanProgress = "failed", isScanning = false) }
            return
        }

        // Step 2: Tell backend scanning has started at this stop
        try {
            apiService.startScanAtStop(routeId, event.stopIndex)
        } catch (_: Exception) { /* Non-fatal */ }

        _state.update { it.copy(scanProgress = "scanning") }

        // Step 3: Wait for scan duration (pipeline is continuously processing)
        delay(SCAN_DURATION_MS)

        // Step 4: Record cooldown
        scanCooldowns[event.stopId] = System.currentTimeMillis()
        _state.update { it.copy(scanProgress = "complete", isScanning = false) }

        // Step 5: Mark stop as completed on backend + advance
        try {
            val resp = apiService.completeStop(routeId, event.stopIndex)
            if (resp.isSuccessful) {
                val body = resp.body()
                val nextIndex = when (val v = body?.get("next_stop_index")) {
                        is Number -> v.toInt()
                        else -> null
                    }
                if (nextIndex != null) {
                    advanceToStop(nextIndex)
                } else {
                    _state.update { it.copy(scanProgress = "route_complete") }
                }
            }
        } catch (_: Exception) { /* Non-fatal */ }
    }

    fun advanceToStop(index: Int) {
        _state.update {
            it.copy(
                currentStopIndex = index,
                scanProgress = "",
                isScanning = false
            )
        }
        // Arrival detector will pick up the new target on next location update
    }

    fun manualCompleteStop(scope: CoroutineScope) {
        scope.launch {
            val routeId = _state.value.activeRoute?.id ?: return@launch
            val stopIndex = _state.value.currentStopIndex
            try {
                val resp = apiService.completeStop(routeId, stopIndex)
                if (resp.isSuccessful) {
                    val body = resp.body()
                    val nextIndex = when (val v = body?.get("next_stop_index")) {
                        is Number -> v.toInt()
                        else -> null
                    }
                    if (nextIndex != null) {
                        advanceToStop(nextIndex)
                    } else {
                        _state.update { it.copy(scanProgress = "route_complete") }
                    }
                }
            } catch (_: Exception) { /* Swallow */ }
        }
    }

    fun skipCurrentStop(scope: CoroutineScope) {
        scope.launch {
            val routeId = _state.value.activeRoute?.id ?: return@launch
            val stopIndex = _state.value.currentStopIndex
            try {
                val resp = apiService.skipStop(routeId, stopIndex)
                if (resp.isSuccessful) {
                    val body = resp.body()
                    val nextIndex = when (val v = body?.get("next_stop_index")) {
                        is Number -> v.toInt()
                        else -> null
                    }
                    if (nextIndex != null) {
                        advanceToStop(nextIndex)
                    }
                }
            } catch (_: Exception) { /* Swallow */ }
        }
    }

    private fun distanceBetween(
        lat1: Double, lng1: Double,
        lat2: Double, lng2: Double
    ): Float = GeoUtils.distanceBetween(lat1, lng1, lat2, lng2)
}
