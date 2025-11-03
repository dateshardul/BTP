using UnityEngine;
using Unity.Netcode;
using System.Collections.Generic;

/// <summary>
/// Manages annotation markers (pins) on the terrain
/// Allows teacher to place, remove, and manage markers
/// All students see the same markers
/// </summary>
public class AnnotationSystem : NetworkBehaviour
{
    [Header("Marker Prefab")]
    [SerializeField] private GameObject markerPinPrefab;  // Prefab with MarkerPin component

    [Header("Marker Settings")]
    [SerializeField] private Transform markerContainer;  // Parent for all markers
    [SerializeField] private int maxMarkers = 50;  // Maximum number of markers allowed
    [SerializeField] private bool autoNumberMarkers = true;

    // Track all spawned markers
    private List<NetworkObject> spawnedMarkers = new List<NetworkObject>();
    private int markerCounter = 0;

    public override void OnNetworkSpawn()
    {
        base.OnNetworkSpawn();

        // Create marker container if not assigned
        if (markerContainer == null)
        {
            GameObject container = new GameObject("MarkerContainer");
            markerContainer = container.transform;
        }
    }

    /// <summary>
    /// Place a marker at specified position
    /// Called by teacher when using Annotate mode
    /// </summary>
    [ServerRpc(RequireOwnership = false)]
    public void PlaceMarkerServerRpc(Vector3 position, ServerRpcParams rpcParams = default)
    {
        if (!IsServer)
        {
            Debug.LogWarning("PlaceMarker can only be called on server");
            return;
        }

        // Check marker limit
        if (spawnedMarkers.Count >= maxMarkers)
        {
            Debug.LogWarning($"Maximum marker limit reached ({maxMarkers})");
            NotifyMarkerLimitClientRpc(rpcParams.Receive.SenderClientId);
            return;
        }

        // Check if prefab is assigned
        if (markerPinPrefab == null)
        {
            Debug.LogError("Marker pin prefab not assigned!");
            return;
        }

        // Spawn marker
        GameObject markerObj = Instantiate(markerPinPrefab, position, Quaternion.identity);
        markerObj.transform.SetParent(markerContainer);

        // Get NetworkObject component
        NetworkObject networkObj = markerObj.GetComponent<NetworkObject>();
        if (networkObj == null)
        {
            Debug.LogError("Marker prefab missing NetworkObject component!");
            Destroy(markerObj);
            return;
        }

        // Spawn on network
        networkObj.Spawn();

        // Configure marker
        MarkerPin markerPin = markerObj.GetComponent<MarkerPin>();
        if (markerPin != null)
        {
            markerCounter++;
            if (autoNumberMarkers)
            {
                markerPin.SetLabelClientRpc($"Pin {markerCounter}");
            }
            markerPin.SetCreatorClientRpc(rpcParams.Receive.SenderClientId);
        }

        // Track spawned marker
        spawnedMarkers.Add(networkObj);

        Debug.Log($"Marker placed at {position} by client {rpcParams.Receive.SenderClientId}");
    }

    /// <summary>
    /// Place marker with custom label
    /// </summary>
    [ServerRpc(RequireOwnership = false)]
    public void PlaceMarkerWithLabelServerRpc(Vector3 position, string label, ServerRpcParams rpcParams = default)
    {
        if (!IsServer) return;

        if (spawnedMarkers.Count >= maxMarkers)
        {
            NotifyMarkerLimitClientRpc(rpcParams.Receive.SenderClientId);
            return;
        }

        if (markerPinPrefab == null) return;

        GameObject markerObj = Instantiate(markerPinPrefab, position, Quaternion.identity);
        markerObj.transform.SetParent(markerContainer);

        NetworkObject networkObj = markerObj.GetComponent<NetworkObject>();
        if (networkObj == null)
        {
            Destroy(markerObj);
            return;
        }

        networkObj.Spawn();

        MarkerPin markerPin = markerObj.GetComponent<MarkerPin>();
        if (markerPin != null)
        {
            markerPin.SetLabelClientRpc(label);
            markerPin.SetCreatorClientRpc(rpcParams.Receive.SenderClientId);
        }

        spawnedMarkers.Add(networkObj);

        Debug.Log($"Labeled marker '{label}' placed at {position}");
    }

    /// <summary>
    /// Remove specific marker
    /// </summary>
    [ServerRpc(RequireOwnership = false)]
    public void RemoveMarkerServerRpc(ulong markerNetworkId)
    {
        if (!IsServer) return;

        // Find marker by network ID
        NetworkObject markerToRemove = null;
        foreach (var marker in spawnedMarkers)
        {
            if (marker != null && marker.NetworkObjectId == markerNetworkId)
            {
                markerToRemove = marker;
                break;
            }
        }

        if (markerToRemove != null)
        {
            spawnedMarkers.Remove(markerToRemove);
            markerToRemove.Despawn();
            Destroy(markerToRemove.gameObject);
            Debug.Log($"Marker {markerNetworkId} removed");
        }
    }

    /// <summary>
    /// Remove all markers
    /// </summary>
    [ServerRpc(RequireOwnership = false)]
    public void RemoveAllMarkersServerRpc()
    {
        if (!IsServer) return;

        foreach (var marker in spawnedMarkers)
        {
            if (marker != null)
            {
                marker.Despawn();
                Destroy(marker.gameObject);
            }
        }

        spawnedMarkers.Clear();
        markerCounter = 0;

        Debug.Log("All markers removed");
    }

    /// <summary>
    /// Remove last placed marker (undo function)
    /// </summary>
    [ServerRpc(RequireOwnership = false)]
    public void RemoveLastMarkerServerRpc()
    {
        if (!IsServer) return;

        if (spawnedMarkers.Count > 0)
        {
            NetworkObject lastMarker = spawnedMarkers[spawnedMarkers.Count - 1];
            if (lastMarker != null)
            {
                spawnedMarkers.RemoveAt(spawnedMarkers.Count - 1);
                lastMarker.Despawn();
                Destroy(lastMarker.gameObject);
                Debug.Log("Last marker removed");
            }
        }
    }

    /// <summary>
    /// Notify client that marker limit was reached
    /// </summary>
    [ClientRpc]
    private void NotifyMarkerLimitClientRpc(ulong clientId)
    {
        if (NetworkManager.Singleton.LocalClientId == clientId)
        {
            Debug.LogWarning($"Cannot place more markers. Limit: {maxMarkers}");
            // Could trigger UI notification here
        }
    }

    /// <summary>
    /// Get marker at position (for selection/editing)
    /// </summary>
    public MarkerPin GetMarkerAtPosition(Vector3 position, float radius = 0.1f)
    {
        foreach (var marker in spawnedMarkers)
        {
            if (marker != null)
            {
                float distance = Vector3.Distance(marker.transform.position, position);
                if (distance < radius)
                {
                    return marker.GetComponent<MarkerPin>();
                }
            }
        }
        return null;
    }

    /// <summary>
    /// Get all current markers
    /// </summary>
    public List<MarkerPin> GetAllMarkers()
    {
        List<MarkerPin> markers = new List<MarkerPin>();
        foreach (var markerObj in spawnedMarkers)
        {
            if (markerObj != null)
            {
                MarkerPin pin = markerObj.GetComponent<MarkerPin>();
                if (pin != null)
                {
                    markers.Add(pin);
                }
            }
        }
        return markers;
    }

    // Public properties
    public int MarkerCount => spawnedMarkers.Count;
    public int MaxMarkers => maxMarkers;
    public bool CanPlaceMarker => spawnedMarkers.Count < maxMarkers;
}