/**
 * Placeholder dashboard shell.
 *
 * Real data will be loaded from `/predictions.json`, written by the Python
 * pipeline into `frontend/public/`. See CLAUDE.md -> "Data contract".
 */

const TEAMS = [
  'Arsenal',
  'Aston Villa',
  'Bournemouth',
  'Brentford',
  'Brighton',
  'Burnley',
  'Chelsea',
  'Crystal Palace',
  'Everton',
  'Fulham',
  'Leeds United',
  'Liverpool',
  'Manchester City',
  'Manchester United',
  'Newcastle United',
  "Nott'm Forest",
  'Sunderland',
  'Tottenham',
  'West Ham',
  'Wolves',
]

const AWARDS = [
  { label: 'Champion', icon: '🏆' },
  { label: 'Top Scorer', icon: '⚽' },
  { label: 'Top Assists', icon: '🅰️' },
  { label: 'Player of the Season', icon: '⭐' },
]

function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/60">
        <div className="mx-auto max-w-5xl px-6 py-8">
          <h1 className="text-3xl font-bold tracking-tight">Premier League Predictor</h1>
          <p className="mt-1 text-sm text-slate-400">
            Predicted final table — awaiting model output
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-10">
        <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {AWARDS.map((award) => (
            <div
              key={award.label}
              className="rounded-lg border border-slate-800 bg-slate-900 p-4"
            >
              <div className="text-2xl">{award.icon}</div>
              <div className="mt-2 text-xs uppercase tracking-wide text-slate-500">
                {award.label}
              </div>
              <div className="mt-1 text-lg font-semibold text-slate-600">TBD</div>
            </div>
          ))}
        </section>

        <section className="mt-10 overflow-x-auto rounded-lg border border-slate-800">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-900 text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-4 py-3 font-medium">#</th>
                <th className="px-4 py-3 font-medium">Team</th>
                <th className="px-4 py-3 text-right font-medium">Pts</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {TEAMS.map((team, i) => (
                <tr key={team} className="hover:bg-slate-900/50">
                  <td className="px-4 py-2.5 tabular-nums text-slate-500">{i + 1}</td>
                  <td className="px-4 py-2.5 font-medium">{team}</td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-slate-600">—</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </main>
    </div>
  )
}

export default App
