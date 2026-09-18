# Supabase authentication

StemScope supports Supabase email/password signup, sign-in and local sign-out.
The existing Gradio interface runs behind a login page when Supabase is configured.
The API verifies Supabase access tokens and separates storage by verified user ID.
No Supabase database tables, storage buckets or service-role key are required.
Audio stays on the StemScope host; only authentication requests go to Supabase.

## Connect your project

1. In your Supabase project, enable the Email authentication provider. Leave email
   confirmation enabled if you want users to verify their addresses. Configure
   the Auth Site URL as `http://127.0.0.1:7860` for this local setup.
2. Copy the project URL and **publishable key** (or legacy **anon** key).
   Do not use a secret/service-role key. StemScope rejects recognized secret keys.
3. From the StemScope directory, copy `.env.example` to `.env.local` and replace
   the placeholders. `.env.local` is ignored by source control. Load it and start:

```sh
set -a
source .env.local
set +a
uv sync --frozen --extra dev
uv run --frozen stemscope
```

Open `http://127.0.0.1:7860`. Create an account, confirm the email if requested,
then sign in. Click **Open StemScope** to reach the full interface. The
**Account / sign out** link at the bottom returns to the account page.
Email confirmation is followed by an explicit password sign-in; the app does not
consume tokens from email URL fragments. Signup requires an 8–128 character
password; Supabase's configured password policy also applies.

The `.env` file is not automatically loaded: source it in each launch shell as
shown. `STEMSCOPE_AUTH_REQUIRED=true` makes startup fail when credentials are
missing. Partial Supabase configuration also fails. Without Supabase variables
and without that flag, the original local, unauthenticated mode remains available.

## API and Docker

The authenticated UI server also hosts `/models`, `/separations` and stem downloads.
For programmatic clients, send `Authorization: Bearer <Supabase access token>`.
Supabase's public API key alone is not a user access token. Browser requests use
an opaque HttpOnly, SameSite=Strict session cookie instead. Every protected request
checks the access token against Supabase's `/auth/v1/user` endpoint.
`/health` remains public and reports no user data.

The standalone `stemscope-api` command supports the same authentication. When
using its browser login pages on port 8000, set
`STEMSCOPE_AUTH_ORIGIN=http://127.0.0.1:8000`. Use the exact configured hostname;
`localhost` and `127.0.0.1` are different origins. Bearer-token API clients do not
need a browser Origin header. Cookie-authenticated writes and account forms must
come from the configured origin.

For the Docker API, rebuild after these code changes, set the origin to the
published port, and pass `.env.local` at runtime:

```sh
docker build --platform linux/amd64 -t stemscope .
docker run --rm --platform linux/amd64 --env-file .env.local \
  -p 127.0.0.1:8000:8000 -v stemscope-storage:/storage stemscope
```

Credentials are not copied into the image. The existing Docker build validation
predates this auth change; the new integration has local automated coverage.

## Isolation and session behavior

Each identity gets `outputs/users/<Supabase-user-id>/`. API downloads resolve only
within that user's directory. Gradio has separate state, queues and upload caches
per user; file serving (including its legacy file URL) and callback file inputs
are restricted to that workspace. Model adapters are shared to reuse weights.
Their inference locks still apply, but different users may run other analysis
work concurrently. This does not create shared public audio storage.

Existing pre-authentication jobs and research artifacts stay in their original
location and are not automatically exposed to new users. In a new workspace,
historical reports and trained Auto policy artifacts may show unavailable; manual
separation, analysis, diagnostics, practice and comparison work normally. Explicit
research-artifact migration is a separate decision.

Browser sessions are kept in server memory for at most one hour or the token's
shorter lifetime. Expiration or a server restart requires signing in again. There
is no automatic refresh. Sign-out invalidates this app's cookie session; it does
not revoke independently issued Bearer tokens or sign the user out of other apps.
Run one server worker. Up to 20 Gradio workspaces and 1,000 active browser sessions
are retained per process; restart clears the in-memory state while files remain.

Password reset, OAuth providers, MFA and CAPTCHA UI are not implemented. Supabase
rate limits apply; the app returns understandable messages for rejected credentials,
rate limits and upstream failures. Public deployment needs HTTPS (which enables
Secure cookies) and deployment-specific capacity/abuse controls.

## Verification

Automated tests use a simulated Supabase HTTP transport; they do not create accounts
or send email. They cover token rejection, upstream failure, cookie handling, logout,
expiry, origin checks, API ownership, Gradio file ownership and uploads/callbacks.
A full Gradio separation workflow runs through authenticated routes with a fixture
separator. Live Supabase sign-in/email delivery still requires project credentials.

References: [Supabase user validation](https://supabase.com/docs/reference/python/auth-getuser),
[Supabase Auth REST contract](https://github.com/supabase/auth/blob/master/openapi.yaml),
and [Gradio external authentication](https://www.gradio.app/guides/sharing-your-app).

Validation on 2026-09-14: **165 tests passed**, including full authenticated Gradio
upload/separation, both Gradio file URL variants, JSON callback file ownership,
missing/expired credentials, logout, origin checks and secret-key rejection.
Lint, formatting and frozen dependency sync passed. No live Supabase account,
confirmation email or authenticated Docker run was exercised without project values.
