import { useState, useRef, useEffect } from 'react'
import { analyzeOutfit } from './api'
import './App.css'

const MAX_FILE_BYTES = 10 * 1024 * 1024 // 10 MB

function Swatch({ label, rgb }) {
  return (
    <div className="swatch">
      <span
        className="swatch-box"
        style={{ backgroundColor: `rgb(${rgb.join(', ')})` }}
      />
      <span className="swatch-label">{label}</span>
    </div>
  )
}

function Row({ label, children }) {
  return (
    <div className="row">
      <span className="row-label">{label}</span>
      <span className="row-value">{children}</span>
    </div>
  )
}

function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [status, setStatus] = useState('idle') // idle | loading | success | error
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [cameraOpen, setCameraOpen] = useState(false)
  const [cameraError, setCameraError] = useState('')

  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)

  function setPickedFile(chosen) {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setFile(chosen)
    setPreviewUrl(URL.createObjectURL(chosen))
    setStatus('idle')
    setResult(null)
    setError('')
  }

  function handleFileChange(event) {
    const chosen = event.target.files[0]
    if (!chosen) return

    if (!chosen.type.startsWith('image/')) {
      setError('Please choose an image file.')
      setStatus('error')
      event.target.value = ''
      return
    }
    if (chosen.size > MAX_FILE_BYTES) {
      setError('That image is too large — please choose one under 10 MB.')
      setStatus('error')
      event.target.value = ''
      return
    }

    setPickedFile(chosen)
  }

  async function openCamera() {
    setCameraError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' },
      })
      streamRef.current = stream
      setCameraOpen(true)
    } catch {
      setCameraError(
        'Could not access the camera. Check your browser permissions.'
      )
    }
  }

  function closeCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    setCameraOpen(false)
  }

  // Attach the stream to the <video> element once it's mounted and open.
  useEffect(() => {
    if (cameraOpen && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current
    }
  }, [cameraOpen])

  // Stop the camera if the component ever unmounts while it's open.
  useEffect(() => {
    return () => streamRef.current?.getTracks().forEach((t) => t.stop())
  }, [])

  function capturePhoto() {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)

    canvas.toBlob((blob) => {
      if (!blob) return
      const captured = new File([blob], 'camera-capture.jpg', {
        type: 'image/jpeg',
      })
      setPickedFile(captured)
      closeCamera()
    }, 'image/jpeg')
  }

  async function handleAnalyze() {
    if (!file) return
    setStatus('loading')
    setError('')
    setResult(null)
    try {
      const data = await analyzeOutfit(file)
      setResult(data)
      setStatus('success')
    } catch (err) {
      setError(err.message)
      setStatus('error')
    }
  }

  const showPhrased =
    result?.llm_phrased_recommendation &&
    result.llm_phrased_recommendation !== result.recommendations.join(' ')

  return (
    <div className="page">
      <main className="app">
        <header className="masthead">
          <h1>Outfit Check</h1>
          <p className="subtitle">
            A full-body photo, read for fit, color, and balance.
          </p>
        </header>

        <section className="upload" aria-label="Provide a photo">
          {cameraOpen ? (
            <div className="camera">
              <video ref={videoRef} autoPlay playsInline muted />
              <canvas ref={canvasRef} hidden />
              <div className="upload-controls">
                <button onClick={capturePhoto}>Capture</button>
                <label className="file-label" onClick={closeCamera}>
                  Cancel
                </label>
              </div>
            </div>
          ) : (
            <>
              {previewUrl ? (
                <img
                  className="preview"
                  src={previewUrl}
                  alt="Selected outfit"
                />
              ) : (
                <div className="upload-empty">
                  <p>No photo selected</p>
                  <p className="upload-hint">
                    Full body, plain background, top and bottom visible
                  </p>
                </div>
              )}

              {cameraError && <p className="status-error">{cameraError}</p>}

              <div className="upload-controls">
                <label className="file-label">
                  {file ? 'Change photo' : 'Choose photo'}
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    hidden
                  />
                </label>
                <label className="file-label" onClick={openCamera}>
                  Use camera
                </label>
                <button
                  onClick={handleAnalyze}
                  disabled={!file || status === 'loading'}
                >
                  {status === 'loading' ? 'Analyzing' : 'Analyze outfit'}
                </button>
              </div>
            </>
          )}
        </section>

        {status === 'loading' && (
          <p className="status-note">Running the model — a few seconds.</p>
        )}
        {status === 'error' && <p className="status-error">{error}</p>}

        {status === 'success' && result && (
          <section className="result">
            <div className="score-block">
              <span className="score">
                {Math.round(result.final_score * 100)}
              </span>
              <span className="score-of">/ 100</span>
            </div>

            <div className="spec-sheet">
              <Row label="Body shape">
                {result.body_shape.replaceAll('_', ' ')}
              </Row>
              <Row label="Color relationship">
                {result.harmony_relationship}
              </Row>
              <Row label="Palette">
                <div className="swatches">
                  <Swatch label="Top" rgb={result.top_color} />
                  <Swatch label="Bottom" rgb={result.bottom_color} />
                </div>
              </Row>
            </div>

            <h2 className="rec-heading">Recommendations</h2>
            <ul className="rec-list">
              {result.recommendations.map((rec, i) => (
                <li key={i}>{rec}</li>
              ))}
            </ul>

            {showPhrased && (
              <p className="rec-phrased">
                {result.llm_phrased_recommendation}
              </p>
            )}
          </section>
        )}
      </main>
    </div>
  )
}

export default App