using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using System.Collections.Generic;
using System.Globalization;
using System.Text.RegularExpressions;

public class UdpLandmarksReceiver : MonoBehaviour
{
    [Header("Network")]
    [Tooltip("same port as Python sender")]
    public int port = 5005;

    [Header("Input coordinates")]
    [Tooltip("check if x and y are normalized (0...1)")]
    public bool inputIsNormalized = true;
    public int imageWidth = 640, imageHeight = 480;
    [Tooltip("flip y-axis (mediapipe vs unity coordinate system)")]
    public bool flipY = true;

    [Header("View")]
    public LineRenderer lineRenderer;
    [Tooltip("overall world scale")]
    public float worldScale = 2.0f;
    [Tooltip("line thickness multiplier")]
    public float lineWidth = 0.02f;

    [Header("Smoothing (optional)")]
    [Range(0f, 1f)] public float emaAlpha = 0.0f; 
    // ─────────────────────────────────────────────────────────────
    UdpClient _client;
    Thread _thread;
    volatile bool _running;

    readonly object _lock = new object();
    List<Vector3> _latest;
    List<Vector3> _smoothed;      // for EMA
    bool _present;

    // MediaPipe 21 points
    static readonly int[] bones = new int[]{
        0,1, 1,2, 2,3, 3,4,
        0,5, 5,6, 6,7, 7,8,
        0,9, 9,10,10,11,11,12,
        0,13,13,14,14,15,15,16,
        0,17,17,18,18,19,19,20
    };

    void Awake()
    {
        if (!lineRenderer) lineRenderer = GetComponent<LineRenderer>();
        if (!lineRenderer) lineRenderer = gameObject.AddComponent<LineRenderer>();
        lineRenderer.positionCount = 0;
        lineRenderer.useWorldSpace = true;
        lineRenderer.widthMultiplier = lineWidth;
    }

    void Start()
    {
        try
        {
            _client = new UdpClient(port);
            _running = true;
            _thread = new Thread(RecvLoop) { IsBackground = true };
            _thread.Start();
            Debug.Log($"UDP listen on {port}");
        }
        catch (Exception e)
        {
            Debug.LogError("UDP open failed: " + e.Message);
        }
    }

    void RecvLoop()
    {
        var ep = new IPEndPoint(IPAddress.Any, port);
        while (_running)
        {
            try
            {
                var data = _client.Receive(ref ep);
                string json = Encoding.UTF8.GetString(data);

                // 21*3 
                var list = TryParseAny(json);

                lock (_lock)
                {
                    _present = (list != null && list.Count >= 21);
                    _latest = _present ? list : null;
                    Debug.Log($"recv len={json.Length}, parsed={(list == null ? 0 : list.Count)}");
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning("RecvLoop error: " + ex.Message);
            }
        }
    }

    List<Vector3> TryParseAny(string s)
    {
        if (string.IsNullOrEmpty(s)) return null;

        s = s.Replace('\'', '\"');

        var ms = Regex.Matches(s, @"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?");
        if (ms.Count < 63) return null;

        var vals = new List<float>(ms.Count);
        foreach (Match m in ms)
            vals.Add(float.Parse(m.Value, CultureInfo.InvariantCulture));


        var pts = new List<Vector3>(21);
        for (int i = 0; i < 63; i += 3)
        {
            float x = vals[i], y = vals[i + 1], z = vals[i + 2];
            if (!inputIsNormalized) { x /= imageWidth; y /= imageHeight; }
            pts.Add(new Vector3(x, y, z));
        }
        return pts;
    }

    void Update()
    {
        List<Vector3> pts = null; bool present;
        lock (_lock) { present = _present; if (_latest != null) pts = new List<Vector3>(_latest); }
        if (!present || pts == null) { lineRenderer.positionCount = 0; return; }

        // smoothing
        if (emaAlpha > 0f)
        {
            if (_smoothed == null || _smoothed.Count != pts.Count)
                _smoothed = new List<Vector3>(pts);
            else
                for (int i = 0; i < pts.Count; i++)
                    _smoothed[i] = emaAlpha * pts[i] + (1f - emaAlpha) * _smoothed[i];
            pts = _smoothed;
        }

        // drawing lines
        lineRenderer.positionCount = bones.Length;
        for (int bi = 0; bi < bones.Length; bi++)
        {
            var p = pts[bones[bi]];

            // Flip Y (MediaPipe 0=top, 1=bottom → Unity 0=bottom, 1=top)
            if (flipY)
                p.y = 1f - p.y;

            var world = new Vector3(
                p.x - 0.5f,
                (p.y - 0.5f),   
                -p.z
            ) * worldScale;

            lineRenderer.SetPosition(bi, world);
        }

        void OnDestroy()
        {
            _running = false;
            try { _client?.Close(); } catch { }
            try { _thread?.Abort(); } catch { }
        }
    }
}
