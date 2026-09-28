import { useState } from 'react'
import { analyzeOutfit } from './api'
import './App.css'

function Swatch({ label, rgb }) {
  return (
    <div className="swatch">
      <div
        className="swatch-box"
        style={{ backgroundColor: `rgb(${rgb.join(', ')})` }}
      />
      <span>{label}</span>
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
    <main className="app">
      <h1>Outfit Check</h1>
      <p className="subtitle">
        Upload a full-body photo and get your outfit scored.
      </p>

      <input type="file" accept="image/*" onChange={handleFileChange} />

      {previewUrl && (
        <img className="preview" src={previewUrl} alt="Your selected outfit" />
      )}

      <button onClick={handleAnalyze} disabled={!file || status === 'loading'}>
        {status === 'loading' ? 'Analyzing...' : 'Analyze outfit'}
      </button>

      {status === 'loading' && (
        <p>Running the model. This can take a few seconds.</p>
      )}
      {status === 'error' && <p className="error">{error}</p>}

      {status === 'success' && result && (
        <section className="result">
          <p className="score">{Math.round(result.final_score * 100)}%</p>
          <p>Overall outfit score</p>

          <p>
            Body shape: <strong>{result.body_shape.replaceAll('_', ' ')}</strong>
          </p>

          <div className="swatches">
            <Swatch label="Top" rgb={result.top_color} />
            <Swatch label="Bottom" rgb={result.bottom_color} />
          </div>
          <p>Color relationship: {result.harmony_relationship}</p>

          <h2>Recommendations</h2>
          <ul>
            {result.recommendations.map((rec, i) => (
              <li key={i}>{rec}</li>
            ))}
          </ul>

          {showPhrased && <p>{result.llm_phrased_recommendation}</p>}
        </section>
      )}
    </main>
  )
}

export default App