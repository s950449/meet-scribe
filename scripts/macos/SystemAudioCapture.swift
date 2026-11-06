import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit

/// Writes 48 kHz mono s16le PCM to stdout.
///
/// Build:
///   swiftc -O -o ~/.local/bin/meetscribe-system-audio scripts/macos/SystemAudioCapture.swift
///
/// The terminal (or the app that launches this binary) needs Screen Recording permission.
/// Python stamps each read with the host clock; this process does not write a WAV header.

final class AudioSink: NSObject, SCStreamOutput, SCStreamDelegate {
    private let output = FileHandle.standardOutput
    private var stream: SCStream?

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        guard let display = content.displays.first else {
            throw CaptureError.noDisplay
        }
        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 48_000
        config.channelCount = 1
        // ScreenCaptureKit still wants a video size even for an audio-only stream.
        config.width = 2
        config.height = 2
        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: .global())
        try await stream.startCapture()
        self.stream = stream
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, CMSampleBufferIsValid(sampleBuffer) else { return }
        guard let format = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbdPointer = CMAudioFormatDescriptionGetStreamBasicDescription(format) else {
            return
        }
        let asbd = asbdPointer.pointee
        var blockBuffer: CMBlockBuffer?
        let bufferListPointer = UnsafeMutablePointer<AudioBufferList>.allocate(capacity: 1)
        defer { bufferListPointer.deallocate() }
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: nil,
            bufferListOut: bufferListPointer,
            bufferListSize: MemoryLayout<AudioBufferList>.size,
            blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil,
            flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
            blockBufferOut: &blockBuffer
        )
        guard status == noErr else { return }
        let buffers = UnsafeMutableAudioBufferListPointer(bufferListPointer)
        guard let first = buffers.first, let data = first.mData, first.mDataByteSize > 0 else { return }
        let channels = max(1, Int(asbd.mChannelsPerFrame))
        let isFloat = asbd.mFormatFlags & kAudioFormatFlagIsFloat != 0
        if isFloat {
            let count = Int(first.mDataByteSize) / MemoryLayout<Float>.size
            let frames = count / channels
            let samples = data.bindMemory(to: Float.self, capacity: count)
            var pcm = [Int16]()
            pcm.reserveCapacity(frames)
            for frame in 0..<frames {
                var mixed: Float = 0
                for channel in 0..<channels {
                    mixed += samples[frame * channels + channel]
                }
                mixed /= Float(channels)
                let clipped = max(-1 as Float, min(1, mixed))
                pcm.append(Int16(clipped * 32767))
            }
            pcm.withUnsafeBytes { raw in
                if let base = raw.baseAddress {
                    output.write(Data(bytes: base, count: raw.count))
                }
            }
        } else {
            output.write(Data(bytes: data, count: Int(first.mDataByteSize)))
        }
    }
}

enum CaptureError: Error, CustomStringConvertible {
    case noDisplay

    var description: String {
        switch self {
        case .noDisplay:
            return "ScreenCaptureKit did not return a display"
        }
    }
}

@main
struct Main {
    static func main() async {
        let sink = AudioSink()
        do {
            try await sink.start()
            dispatchMain()
        } catch {
            fputs("meetscribe-system-audio: \(error)\n", stderr)
            exit(1)
        }
    }
}
