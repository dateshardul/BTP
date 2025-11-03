using UnityEngine;
using Unity.Netcode;
using Unity.Netcode.Transports.UTP;

/// <summary>
/// Manages network connection for host and client
/// Handles starting/stopping network sessions
/// </summary>
public class NetworkConnectionManager : MonoBehaviour
{
    [Header("Network Settings")]
    [SerializeField] private string ipAddress = "127.0.0.1";
    [SerializeField] private ushort port = 7777;

    [Header("References")]
    [SerializeField] private GameObject connectionUI;
    [SerializeField] private GameObject inGameUI;

    private NetworkManager networkManager;
    private UnityTransport transport;

    private void Awake()
    {
        networkManager = NetworkManager.Singleton;
        if (networkManager == null)
        {
            Debug.LogError("NetworkManager not found! Make sure NetworkManager exists in the scene.");
            return;
        }

        transport = networkManager.GetComponent<UnityTransport>();
        if (transport == null)
        {
            Debug.LogError("UnityTransport not found on NetworkManager!");
        }
    }

    private void Start()
    {
        // Subscribe to connection events
        if (networkManager != null)
        {
            networkManager.OnClientConnectedCallback += OnClientConnected;
            networkManager.OnClientDisconnectCallback += OnClientDisconnected;
            networkManager.OnServerStarted += OnServerStarted;
        }

        // Show connection UI at start
        ShowConnectionUI();
    }

    private void OnDestroy()
    {
        // Unsubscribe from events
        if (networkManager != null)
        {
            networkManager.OnClientConnectedCallback -= OnClientConnected;
            networkManager.OnClientDisconnectCallback -= OnClientDisconnected;
            networkManager.OnServerStarted -= OnServerStarted;
        }
    }

    /// <summary>
    /// Start as host (server + client)
    /// </summary>
    public void StartHost()
    {
        if (networkManager == null)
        {
            Debug.LogError("Cannot start host - NetworkManager is null");
            return;
        }

        SetupTransport();

        bool success = networkManager.StartHost();
        if (success)
        {
            Debug.Log("Started as Host");
            HideConnectionUI();
        }
        else
        {
            Debug.LogError("Failed to start as Host");
        }
    }

    /// <summary>
    /// Start as client
    /// </summary>
    public void StartClient()
    {
        if (networkManager == null)
        {
            Debug.LogError("Cannot start client - NetworkManager is null");
            return;
        }

        SetupTransport();

        bool success = networkManager.StartClient();
        if (success)
        {
            Debug.Log($"Attempting to connect to {ipAddress}:{port}");
        }
        else
        {
            Debug.LogError("Failed to start as Client");
        }
    }

    /// <summary>
    /// Start as dedicated server
    /// </summary>
    public void StartServer()
    {
        if (networkManager == null)
        {
            Debug.LogError("Cannot start server - NetworkManager is null");
            return;
        }

        SetupTransport();

        bool success = networkManager.StartServer();
        if (success)
        {
            Debug.Log("Started as Server");
            HideConnectionUI();
        }
        else
        {
            Debug.LogError("Failed to start as Server");
        }
    }

    /// <summary>
    /// Disconnect from network session
    /// </summary>
    public void Disconnect()
    {
        if (networkManager == null) return;

        if (networkManager.IsHost)
        {
            networkManager.Shutdown();
            Debug.Log("Host shut down");
        }
        else if (networkManager.IsClient)
        {
            networkManager.Shutdown();
            Debug.Log("Client disconnected");
        }
        else if (networkManager.IsServer)
        {
            networkManager.Shutdown();
            Debug.Log("Server shut down");
        }

        ShowConnectionUI();
    }

    /// <summary>
    /// Set IP address for connection
    /// </summary>
    public void SetIPAddress(string ip)
    {
        ipAddress = ip;
        Debug.Log($"IP Address set to: {ipAddress}");
    }

    /// <summary>
    /// Set port for connection
    /// </summary>
    public void SetPort(ushort newPort)
    {
        port = newPort;
        Debug.Log($"Port set to: {port}");
    }

    private void SetupTransport()
    {
        if (transport != null)
        {
            transport.ConnectionData.Address = ipAddress;
            transport.ConnectionData.Port = port;
        }
    }

    private void OnServerStarted()
    {
        Debug.Log("Server started successfully");
    }

    private void OnClientConnected(ulong clientId)
    {
        Debug.Log($"Client {clientId} connected");

        // Hide connection UI when successfully connected
        if (clientId == networkManager.LocalClientId)
        {
            HideConnectionUI();
        }
    }

    private void OnClientDisconnected(ulong clientId)
    {
        Debug.Log($"Client {clientId} disconnected");

        // Show connection UI if local client disconnected
        if (clientId == networkManager.LocalClientId)
        {
            ShowConnectionUI();
        }
    }

    private void ShowConnectionUI()
    {
        if (connectionUI != null)
        {
            connectionUI.SetActive(true);
        }

        if (inGameUI != null)
        {
            inGameUI.SetActive(false);
        }
    }

    private void HideConnectionUI()
    {
        if (connectionUI != null)
        {
            connectionUI.SetActive(false);
        }

        if (inGameUI != null)
        {
            inGameUI.SetActive(true);
        }
    }

    // Public getters
    public bool IsConnected => networkManager != null && networkManager.IsConnectedClient;
    public bool IsHost => networkManager != null && networkManager.IsHost;
    public bool IsClient => networkManager != null && networkManager.IsClient;
    public bool IsServer => networkManager != null && networkManager.IsServer;
    public ulong LocalClientId => networkManager != null ? networkManager.LocalClientId : 0;
}