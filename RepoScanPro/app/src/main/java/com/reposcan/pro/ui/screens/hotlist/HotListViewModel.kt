package com.reposcan.pro.ui.screens.hotlist

import androidx.compose.runtime.Immutable
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.reposcan.pro.data.model.HotListEntry
import com.reposcan.pro.domain.usecase.GetHotListEntriesUseCase
import com.reposcan.pro.domain.usecase.ManageHotListUseCase
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

@Immutable
data class HotListState(
    val entries: List<HotListEntry> = emptyList(),
    val isLoading: Boolean = false,
    val error: String? = null,
    val showAddDialog: Boolean = false
)

@HiltViewModel
class HotListViewModel @Inject constructor(
    private val getHotListEntriesUseCase: GetHotListEntriesUseCase,
    private val manageHotListUseCase: ManageHotListUseCase,
    private val savedStateHandle: SavedStateHandle
) : ViewModel() {

    private val _state = MutableStateFlow(HotListState())
    val state: StateFlow<HotListState> = _state.asStateFlow()

    fun loadEntries(isDemoMode: Boolean) {
        viewModelScope.launch {
            _state.update { it.copy(isLoading = true, error = null) }
            getHotListEntriesUseCase(isDemoMode).fold(
                onSuccess = { entries ->
                    _state.update { HotListState(entries = entries) }
                },
                onFailure = { e ->
                    _state.update { HotListState(error = e.message ?: "Failed to load entries") }
                }
            )
        }
    }

    fun deleteEntry(id: String, isDemoMode: Boolean) {
        _state.update { current ->
            current.copy(entries = current.entries.filter { it.id != id })
        }
        viewModelScope.launch {
            manageHotListUseCase.deleteEntry(id, isDemoMode)
        }
    }

    fun addEntry(plateText: String, vehicleInfo: String, isDemoMode: Boolean) {
        viewModelScope.launch {
            manageHotListUseCase.addEntry(plateText, vehicleInfo, isDemoMode).fold(
                onSuccess = { newEntry ->
                    _state.update { current ->
                        current.copy(
                            entries = listOf(newEntry) + current.entries,
                            showAddDialog = false
                        )
                    }
                },
                onFailure = { e ->
                    _state.update { it.copy(error = e.message ?: "Failed to add entry") }
                }
            )
        }
    }

    fun toggleAddDialog() {
        _state.update { it.copy(showAddDialog = !it.showAddDialog) }
    }
}
