/**
 * PRD-266 US-018d — how to get Automatos Go, on /go/pair.
 *
 * Today the app is in testing for invited testers only (D10, D51): TestFlight on iPhone, an install
 * link on Android, and no store listing. GET_AUTOMATOS_GO holds that copy in one place, so the store
 * links replace it when a listing exists. It links nowhere: the invitations carry the links.
 */

/** How to get Automatos Go now. Replace with the store links when a listing exists. */
export const GET_AUTOMATOS_GO = {
  heading: 'Get Automatos Go',
  intro: 'Automatos Go is in testing, for invited testers only.',
  ways: [
    'On iPhone, install it from TestFlight, with the invitation the Automatos team sent you.',
    'On Android, install it from the install link the Automatos team sent you.',
  ],
} as const

/** The 'Get Automatos Go' section. */
export function GetAutomatosGo() {
  return (
    <section aria-labelledby="go-pair-get-the-app" className="space-y-2">
      <h2 id="go-pair-get-the-app" className="text-base font-semibold text-foreground">
        {GET_AUTOMATOS_GO.heading}
      </h2>
      <p className="text-sm text-muted-foreground">{GET_AUTOMATOS_GO.intro}</p>
      <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
        {GET_AUTOMATOS_GO.ways.map((way) => (
          <li key={way}>{way}</li>
        ))}
      </ul>
    </section>
  )
}
