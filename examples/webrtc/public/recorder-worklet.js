class RealtimeRecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0]
    const channel = input && input[0]

    if (!channel || channel.length === 0) {
      return true
    }

    this.port.postMessage(channel.slice())
    return true
  }
}

registerProcessor('realtime-recorder-processor', RealtimeRecorderProcessor)
