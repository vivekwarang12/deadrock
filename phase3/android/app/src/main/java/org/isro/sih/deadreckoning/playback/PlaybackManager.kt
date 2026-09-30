package org.isro.sih.deadreckoning.playback

import kotlinx.coroutines.*
import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader

data class PlaybackRecord(
    val timestampSec: Double,
    val ax: Float, val ay: Float, val az: Float,
    val gx: Float, val gy: Float, val gz: Float,
    val gpsLat: Double?, val gpsLon: Double?,
    val gpsSpeed: Double?, val gpsBearing: Double?
)

/**
 * Dataset Playback Engine for replaying recorded IO-VNBD synchronized test logs.
 * Streams samples in real-time (10 Hz) into the navigation engine.
 */
class PlaybackManager(
    private val onSample: (PlaybackRecord) -> Unit
) {
    private var job: Job? = null
    private var isPlaying = false

    fun startPlayback(records: List<PlaybackRecord>, scope: CoroutineScope) {
        stopPlayback()
        isPlaying = true

        job = scope.launch(Dispatchers.Default) {
            for (record in records) {
                if (!isPlaying) break
                onSample(record)
                delay(100) // 10 Hz real-time delay
            }
        }
    }

    fun stopPlayback() {
        isPlaying = false
        job?.cancel()
        job = null
    }

    companion object {
        fun parseCsvStream(inputStream: InputStream): List<PlaybackRecord> {
            val records = mutableListOf<PlaybackRecord>()
            val reader = BufferedReader(InputStreamReader(inputStream))
            var header: List<String>? = null

            reader.forEachLine { line ->
                val tokens = line.split(",")
                if (header == null) {
                    header = tokens.map { it.trim() }
                } else if (tokens.size >= 6) {
                    try {
                        val t = tokens[0].toDoubleOrNull() ?: 0.0
                        val ax = tokens[1].toFloatOrNull() ?: 0f
                        val ay = tokens[2].toFloatOrNull() ?: 0f
                        val az = tokens[3].toFloatOrNull() ?: 9.81f
                        val gx = tokens[4].toFloatOrNull() ?: 0f
                        val gy = tokens[5].toFloatOrNull() ?: 0f
                        val gz = tokens[6].toFloatOrNull() ?: 0f

                        records.add(
                            PlaybackRecord(
                                timestampSec = t,
                                ax = ax, ay = ay, az = az,
                                gx = gx, gy = gy, gz = gz,
                                gpsLat = tokens.getOrNull(7)?.toDoubleOrNull(),
                                gpsLon = tokens.getOrNull(8)?.toDoubleOrNull(),
                                gpsSpeed = tokens.getOrNull(9)?.toDoubleOrNull(),
                                gpsBearing = tokens.getOrNull(10)?.toDoubleOrNull()
                            )
                        )
                    } catch (e: Exception) {
                        // Skip malformed lines
                    }
                }
            }
            return records
        }
    }
}
