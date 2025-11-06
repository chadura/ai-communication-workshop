import React, { useEffect, useState } from 'react'
import Papa from 'papaparse'

const DEFAULT_API = 'http://localhost:8000/analyze' // change if your API runs elsewhere

function SimpleBar({counts}) {
  // counts: { positive: n, neutral: n, negative: n }
  const max = Math.max(...Object.values(counts), 1)
  return (
    <div className="bar-wrap">
      {Object.entries(counts).map(([k,v]) => (
        <div key={k} className="bar-row">
          <div className="bar-label">{k} ({v})</div>
          <div className="bar-outer">
            <div className="bar-inner" style={{ width: `${(v/max)*100}%` }} />
          </div>
        </div>
      ))}
    </div>
  )
}

export default function App(){
  const [apiUrl, setApiUrl] = useState(DEFAULT_API)
  const [samples, setSamples] = useState([])
  const [inputText, setInputText] = useState('')
  const [singleResult, setSingleResult] = useState(null)
  const [batchResults, setBatchResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [counts, setCounts] = useState({positive:0, neutral:0, negative:0})
  const [progress, setProgress] = useState({done:0,total:0})
  const [errorMsg, setErrorMsg] = useState(null)

  useEffect(() => {
    // load sample messages from public assets
    fetch('/assets/sample_messages.json').then(r => {
      if (!r.ok) return []
      return r.json()
    }).then(json => {
      // json is array of objects with text
      if (Array.isArray(json)) {
        const texts = json.map(item => item.text ? item.text : String(item))
        setSamples(texts)
      } else {
        setSamples([])
      }
    }).catch(err => {
      console.warn('Could not load sample messages:', err)
      setSamples([])
    })
  }, [])

  async function analyzeSingle(){
    setErrorMsg(null)
    setSingleResult(null)
    if (!inputText || !inputText.trim()){
      setErrorMsg('Please enter text to analyze.')
      return
    }
    try {
      setLoading(true)
      const res = await fetch(apiUrl, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: inputText})
      })
      if (!res.ok){
        const txt = await res.text()
        throw new Error(`API error: ${res.status} ${txt}`)
      }
      const json = await res.json()
      setSingleResult(json)
    } catch (e){
      setErrorMsg(String(e))
    } finally {
      setLoading(false)
    }
  }

  async function analyzeBatchFromArray(texts){
    setErrorMsg(null)
    setBatchResults([])
    setCounts({positive:0, neutral:0, negative:0})
    setProgress({done:0, total: texts.length})
    const out = []
    for (let i=0;i<texts.length;i++){
      const t = texts[i]
      try {
        const res = await fetch(apiUrl, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({text: t})
        })
        if (!res.ok){
          // fallback: mark as neutral
          out.push({text: t, prediction: 'neutral', proba: {}, error: `HTTP ${res.status}`})
        } else {
          const j = await res.json()
          out.push({text: t, prediction: j.prediction, proba: j.proba || {}, error: null})
        }
      } catch (e){
        out.push({text: t, prediction: 'neutral', proba: {}, error: String(e)})
      }
      setProgress(prev => ({...prev, done: prev.done + 1}))
    }
    setBatchResults(out)
    // aggregate counts
    const c = out.reduce((acc, cur) => {
      const k = cur.prediction || 'neutral'
      acc[k] = (acc[k] || 0) + 1
      return acc
    }, {})
    setCounts({positive: c.positive || 0, neutral: c.neutral || 0, negative: c.negative || 0})
  }

  function handleCSVUpload(file){
    setErrorMsg(null)
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: function(results){
        // look for 'text' column
        const data = results.data
        if (!data || data.length === 0){
          setErrorMsg('CSV looks empty or invalid.')
          return
        }
        if (!Object.keys(data[0]).includes('text')){
          setErrorMsg('CSV must contain a `text` column.')
          return
        }
        const texts = data.map(row => String(row.text))
        analyzeBatchFromArray(texts)
      },
      error: function(err){
        setErrorMsg(String(err))
      }
    })
  }

  function analyzeAllSamples(){
    if (!samples || samples.length===0){
      setErrorMsg('No sample messages loaded.')
      return
    }
    analyzeBatchFromArray(samples)
  }

  return (
    <div className="app">
      <header className="header">
        <h1>AI Communication — React Dashboard</h1>
        <div className="header-right">
          <label>API URL: </label>
          <input className="api-input" value={apiUrl} onChange={(e)=>setApiUrl(e.target.value)} />
        </div>
      </header>

      <main className="main-grid">
        <section className="card">
          <h2>Single Message</h2>
          <select onChange={(e)=>setInputText(e.target.value)} value={inputText}>
            <option value="">-- choose a sample or type below --</option>
            {samples.map((s,idx) => <option key={idx} value={s}>{s.length>80? s.slice(0,80)+'...': s}</option>)}
          </select>
          <textarea value={inputText} onChange={(e)=>setInputText(e.target.value)} placeholder="Type message here..." />
          <div className="actions">
            <button onClick={analyzeSingle} disabled={loading}>Analyze</button>
          </div>
          {errorMsg && <div className="error">{errorMsg}</div>}
          {singleResult && (
            <div className="result">
              <h3>Result: {singleResult.prediction}</h3>
              <pre className="proba">{JSON.stringify(singleResult.proba, null, 2)}</pre>
            </div>
          )}
        </section>

        <section className="card">
          <h2>Batch Analysis</h2>
          <div className="batch-actions">
            <button onClick={analyzeAllSamples} disabled={loading || samples.length===0}>Analyze All Samples ({samples.length})</button>
            <label className="upload-label">
              Upload CSV (requires `text` column)
              <input type="file" accept=".csv" onChange={(e)=> {
                if (e.target.files && e.target.files[0]) handleCSVUpload(e.target.files[0])
              }} />
            </label>
          </div>

          <div className="progress">
            {progress.total>0 && <div>Progress: {progress.done}/{progress.total}</div>}
          </div>

          <div className="summary">
            <h4>Summary</h4>
            <SimpleBar counts={counts} />
          </div>

        </section>

        <section className="card card-wide">
          <h2>Results</h2>
          <div className="results-table">
            <table>
              <thead>
                <tr><th>#</th><th>Message</th><th>Prediction</th><th>Error</th></tr>
              </thead>
              <tbody>
                {batchResults.map((r,i) => (
                  <tr key={i}>
                    <td>{i+1}</td>
                    <td className="msg-cell">{r.text}</td>
                    <td>{r.prediction}</td>
                    <td>{r.error ? String(r.error).slice(0,80) : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </main>

      <footer className="footer">
        <small>Workshop demo — not for production use. Default API: {DEFAULT_API}</small>
      </footer>
    </div>
  )
}
