package com.reposcan.pro.ui.screens.map

import androidx.compose.runtime.Immutable
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.reposcan.pro.data.location.LocationProvider
import com.reposcan.pro.data.model.Detection
import com.reposcan.pro.data.model.RouteResponse
import com.reposcan.pro.data.model.RouteStopResponse
import com.reposcan.pro.data.remote.ApiService
import com.reposcan.pro.domain.navigation.AutoScanController
import com.reposcan.pro.domain.usecase.GetDetectionsUseCase
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

@Immutable
data class MapState(
    // GPS
    val userLat: Double = 0.0,
    val userLng: Double = 0.0,
    val userHeading: Float = 0f,
    val userSpeed: Float = 0f,
    val gpsAccuracy: Float = Float.MAX_VALUE,
    val gpsActive: Boolean = false,
    val hasLocationPermission: Boolean = false,

    // Route
    val activeRoute: RouteResponse? = null,
    val currentStopIndex: Int = 0,
    val isLoadingRoute: Boolean = false,

    // Navigation
    val autoScanEnabled: Boolean = false,
    val isAutoScanning: Boolean = false,
    val scanProgress: String = "",
    val distanceToTargetM: Float = Float.MAX_VALUE,
    val isApproaching: Boolean = false,

    // Detections
    val detections: List<Detection> = emptyList(),
    val isLoading: Boolean = false,
    val error: String? = null
) {
    val currentStop: RouteStopResponse?
        get() = activeRoute?.stops?.getOrNull(currentStopIndex)

    val completedStops: Int
        get() = activeRoute?.stops?.count { it.status in listOf("completed", "skipped") } ?: 0

    val remainingStops: Int
        get() = activeRoute?.stops?.count { it.status == "pending" } ?: 0

    val distanceText: String
        get() = when {
            distanceToTargetM == Float.MAX_VALUE -> ""
            distanceToTargetM >= 1000 -> "%.1f km".format(distanceToTargetM / 1000f)
            else -> "%.0f m".format(distanceToTargetM)
        }
}

@HiltViewModel
class MapViewModel @Inject constructor(
    private val getDetectionsUseCase: GetDetectionsUseCase,
    private val locationProvider: LocationProvider,
    private val autoScanController: AutoScanController,
    private val apiService: ApiService,
    private val savedStateHandle: SavedStateHandle
) : ViewModel() {

    private val _state = MutableStateFlow(MapState())
    val state: StateFlow<MapState> = _state.asStateFlow()

    init {
        // Observe GPS location updates
        viewModelScope.launch {
            locationProvider.location.collect { loc ->
                _state.update {
                    it.copy(
                        userLat = loc.latitude,
                        userLng = loc.longitude,
                        userHeading = loc.heading,
                        userSpeed = loc.speed,
                        gpsAccuracy = loc.accuracy,
                        gpsActive = loc.isActive
                    )
                }
            }
        }

        // Observe auto-scan navigation state
        viewModelScope.launch {
            autoScanController.state.collect { navState ->
                _state.update {
                    it.copy(
                        activeRoute = navState.activeRoute ?: it.activeRoute,
                        currentStopIndex = navState.currentStopIndex,
                        autoScanEnabled = navState.autoScanEnabled,
                        isAutoScanning = navState.isScanning,
                        scanProgress = navState.scanProgress,
                        distanceToTargetM = navState.distanceToTargetM,
                        isApproaching = navState.isApproaching
                    )
                }
            }
        }
    }

    fun onLocationPermissionGranted() {
        _state.update { it.copy(hasLocationPermission = true) }
        locationProvider.startUpdates()
    }

    fun loadDetections(isDemoMode: Boolean) {
        viewModelScope.launch {
            _state.update { it.copy(isLoading = true, error = null) }
            getDetectionsUseCase(isDemoMode).fold(
                onSuccess = { list ->
                    val withCoords = list.items.filter {
                        it.latitude != null && it.longitude != null
                    }
                    _state.update { it.copy(detections = withCoords, isLoading = false) }
                },
                onFailure = { e ->
                    _state.update {
                        it.copy(
                            error = e.message ?: "Failed to load detections",
                            isLoading = false
                        )
                    }
                }
            )
        }
    }

    fun loadActiveRoute() {
        viewModelScope.launch {
            _state.update { it.copy(isLoadingRoute = true) }
            try {
                val resp = apiService.getActiveRoute()
                if (resp.isSuccessful) {
                    val route = resp.body() ?: run {
                        _state.update { it.copy(isLoadingRoute = false) }
                        return@launch
                    }
                    _state.update { it.copy(activeRoute = route, isLoadingRoute = false) }
                    autoScanController.loadRoute(route)
                } else {
                    _state.update { it.copy(isLoadingRoute = false) }
                }
            } catch (_: Exception) {
                _state.update { it.copy(isLoadingRoute = false) }
            }
        }
    }

    fun startNavigation() {
        autoScanController.startNavigation(viewModelScope)
    }

    fun stopNavigation() {
        autoScanController.stopNavigation()
    }

    fun toggleAutoScan() {
        val current = _state.value.autoScanEnabled
        autoScanController.setAutoScanEnabled(!current)
    }

    fun completeCurrentStop() {
        autoScanController.manualCompleteStop(viewModelScope)
    }

    fun skipCurrentStop() {
        autoScanController.skipCurrentStop(viewModelScope)
    }

    override fun onCleared() {
        super.onCleared()
        autoScanController.stopNavigation()
    }
}
