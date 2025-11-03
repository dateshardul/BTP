using UnityEngine;
using Unity.Netcode;
using System.Collections.Generic;

/// <summary>
/// Manages player spawn positions around the terrain
/// Spawns players in a circle around the terrain for optimal viewing
/// </summary>
public class PlayerSpawnManager : NetworkBehaviour
{
    [Header("Spawn Settings")]
    [SerializeField] private Transform terrainCenter;
    [SerializeField] private float spawnRadius = 2f;
    [SerializeField] private float spawnHeight = 1.6f; // Average human eye height in VR

    [Header("Spawn Angles (degrees)")]
    [SerializeField] private float[] spawnAngles = { 0f, 90f, 180f, 270f }; // N, E, S, W positions

    private Dictionary<ulong, int> playerSpawnIndices = new Dictionary<ulong, int>();
    private int nextSpawnIndex = 0;

    public override void OnNetworkSpawn()
    {
        base.OnNetworkSpawn();

        if (IsServer)
        {
            NetworkManager.Singleton.OnClientConnectedCallback += OnClientConnected;
            NetworkManager.Singleton.OnClientDisconnectCallback += OnClientDisconnected;
        }
    }

    public override void OnNetworkDespawn()
    {
        if (IsServer)
        {
            if (NetworkManager.Singleton != null)
            {
                NetworkManager.Singleton.OnClientConnectedCallback -= OnClientConnected;
                NetworkManager.Singleton.OnClientDisconnectCallback -= OnClientDisconnected;
            }
        }

        base.OnNetworkDespawn();
    }

    private void OnClientConnected(ulong clientId)
    {
        Debug.Log($"Player {clientId} connected, assigning spawn position");
        AssignSpawnPosition(clientId);
    }

    private void OnClientDisconnected(ulong clientId)
    {
        Debug.Log($"Player {clientId} disconnected, freeing spawn position");
        FreeSpawnPosition(clientId);
    }

    /// <summary>
    /// Assign a spawn position to a newly connected client
    /// </summary>
    private void AssignSpawnPosition(ulong clientId)
    {
        if (!IsServer) return;

        // Assign the next available spawn index
        playerSpawnIndices[clientId] = nextSpawnIndex;

        // Get the player's NetworkObject
        if (NetworkManager.Singleton.ConnectedClients.TryGetValue(clientId, out var client))
        {
            if (client.PlayerObject != null)
            {
                // Calculate and set spawn position
                Vector3 spawnPosition = GetSpawnPosition(nextSpawnIndex);
                Quaternion spawnRotation = GetSpawnRotation(nextSpawnIndex);

                client.PlayerObject.transform.position = spawnPosition;
                client.PlayerObject.transform.rotation = spawnRotation;

                Debug.Log($"Player {clientId} spawned at position {spawnPosition} (index {nextSpawnIndex})");
            }
        }

        // Increment spawn index for next player
        nextSpawnIndex = (nextSpawnIndex + 1) % spawnAngles.Length;
    }

    /// <summary>
    /// Free up a spawn position when a client disconnects
    /// </summary>
    private void FreeSpawnPosition(ulong clientId)
    {
        if (playerSpawnIndices.ContainsKey(clientId))
        {
            playerSpawnIndices.Remove(clientId);
        }
    }

    /// <summary>
    /// Calculate spawn position based on index
    /// </summary>
    private Vector3 GetSpawnPosition(int index)
    {
        if (terrainCenter == null)
        {
            Debug.LogWarning("Terrain center not assigned, using world origin");
            terrainCenter = transform;
        }

        // Get angle for this spawn index
        float angleInDegrees = spawnAngles[index % spawnAngles.Length];
        float angleInRadians = angleInDegrees * Mathf.Deg2Rad;

        // Calculate position on circle around terrain
        Vector3 centerPos = terrainCenter.position;
        float x = centerPos.x + spawnRadius * Mathf.Cos(angleInRadians);
        float z = centerPos.z + spawnRadius * Mathf.Sin(angleInRadians);

        return new Vector3(x, spawnHeight, z);
    }

    /// <summary>
    /// Calculate spawn rotation (looking at terrain center)
    /// </summary>
    private Quaternion GetSpawnRotation(int index)
    {
        if (terrainCenter == null)
        {
            return Quaternion.identity;
        }

        // Calculate direction to look at terrain
        Vector3 spawnPos = GetSpawnPosition(index);
        Vector3 directionToTerrain = terrainCenter.position - spawnPos;
        directionToTerrain.y = 0; // Keep rotation horizontal

        if (directionToTerrain.magnitude > 0.01f)
        {
            return Quaternion.LookRotation(directionToTerrain);
        }

        return Quaternion.identity;
    }

    /// <summary>
    /// Manually spawn a player at a specific position
    /// </summary>
    public void SpawnPlayerAtPosition(ulong clientId, Vector3 position, Quaternion rotation)
    {
        if (!IsServer)
        {
            Debug.LogWarning("SpawnPlayerAtPosition can only be called on the server");
            return;
        }

        if (NetworkManager.Singleton.ConnectedClients.TryGetValue(clientId, out var client))
        {
            if (client.PlayerObject != null)
            {
                client.PlayerObject.transform.position = position;
                client.PlayerObject.transform.rotation = rotation;
                Debug.Log($"Player {clientId} manually spawned at {position}");
            }
        }
    }

    /// <summary>
    /// Get spawn position for debugging/visualization
    /// </summary>
    public Vector3 GetSpawnPositionForIndex(int index)
    {
        return GetSpawnPosition(index);
    }

    // Visualize spawn positions in editor
    private void OnDrawGizmosSelected()
    {
        if (terrainCenter == null) return;

        Gizmos.color = Color.green;

        // Draw spawn positions
        for (int i = 0; i < spawnAngles.Length; i++)
        {
            Vector3 spawnPos = GetSpawnPosition(i);
            Gizmos.DrawWireSphere(spawnPos, 0.2f);

            // Draw line to terrain center
            Gizmos.color = Color.yellow;
            Gizmos.DrawLine(spawnPos, terrainCenter.position);
            Gizmos.color = Color.green;
        }

        // Draw spawn radius circle
        Gizmos.color = Color.cyan;
        DrawCircle(terrainCenter.position, spawnRadius, 32);
    }

    private void DrawCircle(Vector3 center, float radius, int segments)
    {
        float angleStep = 360f / segments;
        Vector3 previousPoint = center + new Vector3(radius, 0, 0);

        for (int i = 1; i <= segments; i++)
        {
            float angle = i * angleStep * Mathf.Deg2Rad;
            Vector3 newPoint = center + new Vector3(
                radius * Mathf.Cos(angle),
                0,
                radius * Mathf.Sin(angle)
            );

            Gizmos.DrawLine(previousPoint, newPoint);
            previousPoint = newPoint;
        }
    }
}