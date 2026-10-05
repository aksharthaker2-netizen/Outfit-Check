import { useState } from 'react'
import { analyzeOutfit } from './api'
import './App.css'

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

  function handleFileChange(event) {
    const chosen = event.target.files[0]
    if (!chosen) return
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setFile(chosen)
    setPreviewUrl(URL.createObjectURL(chosen))
    setStatus('idle')
    setResult(null)
    setError('')
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

        <section className="upload" aria-label="Upload a photo">
          {previewUrl ? (
            <img className="preview" src={previewUrl} alt="Selected outfit" />
          ) : (
            <div className="upload-empty">
              <p>No photo selected</p>
              <p className="upload-hint">
                Full body, plain background, top and bottom visible
              </p>
            </div>
          )}

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
            <button
              onClick={handleAnalyze}
              disabled={!file || status === 'loading'}
            >
              {status === 'loading' ? 'Analyzing' : 'Analyze outfit'}
            </button>
          </div>
        </section>

        {status === 'loading' && (
          <p className="status-note">Running the model — a few seconds.</p>
        )}
        {status === 'error' && <p className="status-error">{error}</p>}

        {status === 'success' && result && (
          <section className="result">
            <div className="score-block">
              <span className="score">{Math.round(result.final_score * 100)}</span>
              <span className="score-of">/ 100</span>
            </div>

            <div className="spec-sheet">
              <Row label="Body shape">
                {result.body_shape.replaceAll('_', ' ')}
              </Row>
              <Row label="Color relationship">{result.harmony_relationship}</Row>
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
              <p className="rec-phrased">{result.llm_phrased_recommendation}</p>
            )}
          </section>
        )}
      </main>
    </div>
  )
}

export default App