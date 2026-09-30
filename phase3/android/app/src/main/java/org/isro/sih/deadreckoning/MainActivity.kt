package org.isro.sih.deadreckoning

import android.content.Context
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import org.isro.sih.deadreckoning.engine.NavigationEngine
import org.isro.sih.deadreckoning.engine.NavState
import org.isro.sih.deadreckoning.playback.PlaybackManager
import org.isro.sih.deadreckoning.playback.PlaybackRecord
import org.osmdroid.config.Configuration
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Polyline
import java.util.Locale

class MainActivity : AppCompatActivity(), SensorEventListener {

    private lateinit var engine: NavigationEngine
    private lateinit var playbackManager: PlaybackManager
    private lateinit var sensorManager: SensorManager

    // UI Elements
    private lateinit var tvModeBadge: TextView
    private lateinit var tvPlaybackBanner: TextView
    private lateinit var tvSpeed: TextView
    private lateinit var tvHeading: TextView
    private lateinit var tvDrift: TextView
    private lateinit var tvConfidence: TextView
    private lateinit var tvMatchedRoad: TextView
    private lateinit var btnToggleSource: Button
    private lateinit var btnSimulateOutage: Button
    private lateinit var mapView: MapView

    // Map Overlays
    private val trajectoryPolyline = Polyline()
    private var isLiveSensorMode = false
    private var isOutageActive = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Configuration.getInstance().load(this, getPreferences(Context.MODE_PRIVATE))
        setContentView(R.layout.activity_main)

        engine = NavigationEngine(this)
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager

        initViews()
        initMap()
        setupListeners()
        startDemoPlayback()
    }

    private fun initViews() {
        tvModeBadge = findViewById(R.id.tvModeBadge)
        tvPlaybackBanner = findViewById(R.id.tvPlaybackBanner)
        tvSpeed = findViewById(R.id.tvSpeed)
        tvHeading = findViewById(R.id.tvHeading)
        tvDrift = findViewById(R.id.tvDrift)
        tvConfidence = findViewById(R.id.tvConfidence)
        tvMatchedRoad = findViewById(R.id.tvMatchedRoad)
        btnToggleSource = findViewById(R.id.btnToggleSource)
        btnSimulateOutage = findViewById(R.id.btnSimulateOutage)
        mapView = findViewById(R.id.mapView)
    }

    private fun initMap() {
        mapView.setMultiTouchControls(true)
        val mapController = mapView.controller
        mapController.setZoom(17.5)
        val startPoint = GeoPoint(52.3850, -1.5600)
        mapController.setCenter(startPoint)

        trajectoryPolyline.outlinePaint.color = Color.parseColor("#38BDF8")
        trajectoryPolyline.outlinePaint.strokeWidth = 8.0f
        mapView.overlays.add(trajectoryPolyline)
    }

    private fun setupListeners() {
        btnToggleSource.setOnClickListener {
            isLiveSensorMode = !isLiveSensorMode
            if (isLiveSensorMode) {
                playbackManager.stopPlayback()
                registerSensors()
                tvPlaybackBanner.visibility = View.GONE
                btnToggleSource.text = "Switch: Dataset Playback"
            } else {
                unregisterSensors()
                tvPlaybackBanner.visibility = View.VISIBLE
                btnToggleSource.text = "Switch: Live Sensors"
                startDemoPlayback()
            }
        }

        btnSimulateOutage.setOnClickListener {
            isOutageActive = !isOutageActive
            engine.setSimulatedOutage(isOutageActive)
            if (isOutageActive) {
                btnSimulateOutage.text = "Restore GNSS Lock"
                btnSimulateOutage.setBackgroundColor(Color.parseColor("#16A34A"))
            } else {
                btnSimulateOutage.text = "Simulate GNSS Loss"
                btnSimulateOutage.setBackgroundColor(Color.parseColor("#DC2626"))
            }
        }
    }

    private fun startDemoPlayback() {
        // Generate a 10Hz synthetic stream along Kenilworth Road
        val records = mutableListOf<PlaybackRecord>()
        for (i in 0 until 500) {
            val t = i * 0.1
            records.add(
                PlaybackRecord(
                    timestampSec = t,
                    ax = 0.1f, ay = 0.0f, az = 9.81f,
                    gx = 0.0f, gy = 0.0f, gz = 0.02f,
                    gpsLat = 52.3850 + (i * 0.00005),
                    gpsLon = -1.5600 + (i * 0.00008),
                    gpsSpeed = 12.5,
                    gpsBearing = 45.0
                )
            )
        }

        playbackManager = PlaybackManager { record ->
            val state = engine.processImu(
                record.timestampSec,
                record.ax, record.ay, record.az,
                record.gx, record.gy, record.gz
            )
            if (record.gpsLat != null && record.gpsLon != null) {
                engine.processGnss(record.gpsLat, record.gpsLon, record.gpsSpeed ?: 0.0, record.gpsBearing ?: 0.0)
            }
            runOnUiThread { updateUI(state) }
        }

        playbackManager.startPlayback(records, lifecycleScope)
    }

    private fun registerSensors() {
        val accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        accel?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        gyro?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
    }

    private fun unregisterSensors() {
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (!isLiveSensorMode || event == null) return

        val t = System.currentTimeMillis() / 1000.0
        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            val state = engine.processImu(t, event.values[0], event.values[1], event.values[2], 0f, 0f, 0f)
            updateUI(state)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun updateUI(state: NavState) {
        tvSpeed.text = String.format(Locale.US, "%.1f", state.speedMs)
        tvHeading.text = String.format(Locale.US, "%.0f°", state.headingDeg)
        tvDrift.text = String.format(Locale.US, "%.1f%%", state.driftPercent)
        tvConfidence.text = String.format(Locale.US, "%.0f%%", state.confidenceScore * 100.0)
        tvMatchedRoad.text = "Road: ${state.matchedRoadName ?: "Analyzing..."}"

        if (state.mode == "GNSS-AIDED-INS") {
            tvModeBadge.text = "GNSS-AIDED"
            tvModeBadge.setBackgroundColor(Color.parseColor("#059669"))
        } else {
            tvModeBadge.text = "PURE-DR"
            tvModeBadge.setBackgroundColor(Color.parseColor("#D97706"))
        }

        val point = GeoPoint(state.latitudeDeg, state.longitudeDeg)
        trajectoryPolyline.addPoint(point)
        mapView.controller.animateTo(point)
        mapView.invalidate()
    }

    override fun onDestroy() {
        super.onDestroy()
        playbackManager.stopPlayback()
        unregisterSensors()
    }
}
