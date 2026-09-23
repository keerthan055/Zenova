"""User-facing Authentication HTML Web Pages (/login, /register, /forgot-password, /reset-password, /verify-email)."""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["Authentication Web Pages"])

AUTH_PAGE_CSS = """
    :root {
        --bg-primary: #f8fafc;
        --bg-surface: #ffffff;
        --text-main: #0f172a;
        --text-muted: #64748b;
        --primary: #0284c7;
        --primary-hover: #0369a1;
        --primary-light: #e0f2fe;
        --accent: #0d9488;
        --border: #e2e8f0;
        --danger: #ef4444;
        --danger-light: #fee2e2;
        --warning: #f59e0b;
        --warning-light: #fef3c7;
        --success: #10b981;
        --success-light: #d1fae5;
        --radius: 12px;
        --shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    }
    [data-theme="dark"] {
        --bg-primary: #0b0f19;
        --bg-surface: #131b2e;
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --primary: #38bdf8;
        --primary-hover: #0ea5e9;
        --primary-light: #1e293b;
        --accent: #2dd4bf;
        --border: #1e293b;
        --danger-light: #450a0a;
        --warning-light: #451a03;
        --success-light: #064e3b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    body {
        background-color: var(--bg-primary);
        color: var(--text-main);
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 24px 16px;
    }
    .auth-card {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        width: 100%;
        max-width: 440px;
        padding: 32px 28px;
    }
    .brand-header {
        text-align: center;
        margin-bottom: 24px;
    }
    .logo-badge {
        display: inline-block;
        background: linear-gradient(135deg, var(--accent), var(--primary));
        color: white;
        font-weight: 800;
        font-size: 20px;
        padding: 8px 16px;
        border-radius: 8px;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
    }
    .brand-header h1 {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .brand-header p {
        font-size: 13px;
        color: var(--text-muted);
    }
    .form-group {
        margin-bottom: 16px;
    }
    .form-group label {
        display: block;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .input-wrapper {
        position: relative;
    }
    .form-control {
        width: 100%;
        padding: 10px 14px;
        border: 1px solid var(--border);
        border-radius: 8px;
        font-size: 14px;
        background: var(--bg-primary);
        color: var(--text-main);
        outline: none;
        transition: border-color 0.15s;
    }
    .form-control:focus {
        border-color: var(--primary);
    }
    .password-toggle-btn {
        position: absolute;
        right: 12px;
        top: 50%;
        transform: translateY(-50%);
        background: none;
        border: none;
        color: var(--text-muted);
        cursor: pointer;
        font-size: 13px;
        padding: 2px;
    }
    .btn-submit {
        width: 100%;
        padding: 12px;
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.15s;
        margin-top: 8px;
    }
    .btn-submit:hover:not(:disabled) {
        background: var(--primary-hover);
    }
    .btn-submit:disabled {
        opacity: 0.6;
        cursor: not-allowed;
    }
    .alert-banner {
        display: none;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 13px;
        margin-bottom: 16px;
        line-height: 1.4;
    }
    .alert-danger {
        background: var(--danger-light);
        border: 1px solid var(--danger);
        color: #b91c1c;
    }
    [data-theme="dark"] .alert-danger { color: #fca5a5; }
    .alert-success {
        background: var(--success-light);
        border: 1px solid var(--success);
        color: #047857;
    }
    [data-theme="dark"] .alert-success { color: #6ee7b7; }
    .auth-links {
        margin-top: 20px;
        text-align: center;
        font-size: 13px;
        color: var(--text-muted);
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .auth-links a {
        color: var(--primary);
        text-decoration: none;
        font-weight: 600;
    }
    .auth-links a:hover {
        text-decoration: underline;
    }
    .disclaimer-box {
        margin-top: 20px;
        padding: 10px;
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: 8px;
        font-size: 11px;
        color: var(--text-muted);
        text-align: center;
        line-height: 1.4;
    }
    .theme-toggle-corner {
        position: fixed;
        top: 16px;
        right: 16px;
        background: none;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 12px;
        color: var(--text-muted);
        cursor: pointer;
    }
"""


@router.get("/login", response_class=HTMLResponse)
async def login_page() -> HTMLResponse:
    """Serve the ZENOVA Login Page."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZENOVA — Sign In</title>
    <style>{AUTH_PAGE_CSS}</style>
</head>
<body>
    <button class="theme-toggle-corner" onclick="toggleTheme()">?? Theme</button>
    <main class="auth-card" role="main">
        <div class="brand-header">
            <div class="logo-badge">ZENOVA</div>
            <h1>Welcome Back</h1>
            <p>Sign in to your supportive companion space</p>
        </div>

        <div id="alertBox" class="alert-banner" role="alert"></div>

        <form id="loginForm" onsubmit="handleLogin(event)">
            <div class="form-group">
                <label for="email">Email Address</label>
                <input id="email" type="email" class="form-control" required autocomplete="email" placeholder="name@example.com">
            </div>

            <div class="form-group">
                <label for="password">Password</label>
                <div class="input-wrapper">
                    <input id="password" type="password" class="form-control" required autocomplete="current-password" placeholder="Enter your password">
                    <button type="button" class="password-toggle-btn" onclick="togglePassword('password')">Show</button>
                </div>
            </div>

            <div class="form-group" style="display: flex; justify-content: space-between; align-items: center; font-size: 12px;">
                <label style="display: flex; align-items: center; gap: 6px; cursor: pointer; font-weight: normal;">
                    <input id="rememberMe" type="checkbox"> Remember this device
                </label>
                <a href="/forgot-password" style="color: var(--primary); text-decoration: none;">Forgot password?</a>
            </div>

            <button id="submitBtn" type="submit" class="btn-submit">Sign In</button>
        </form>

        <div class="auth-links">
            <div>Don't have an account? <a href="/register">Create one here</a></div>
            <div>Need help? <a href="https://988lifeline.org" target="_blank" rel="noopener">24/7 Crisis Support (988)</a></div>
        </div>

        <div class="disclaimer-box">
            ZENOVA is an AI emotional support companion, not a healthcare provider or crisis emergency service.
        </div>
    </main>

    <script>
        function toggleTheme() {{
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            localStorage.setItem('zenova-theme', next);
        }}
        const savedTheme = localStorage.getItem('zenova-theme') || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
        document.documentElement.setAttribute('data-theme', savedTheme);

        function togglePassword(id) {{
            const input = document.getElementById(id);
            const btn = event.target;
            if (input.type === 'password') {{
                input.type = 'text';
                btn.textContent = 'Hide';
            }} else {{
                input.type = 'password';
                btn.textContent = 'Show';
            }}
        }}

        async function handleLogin(e) {{
            e.preventDefault();
            const email = document.getElementById('email').value.trim();
            const password = document.getElementById('password').value;
            const rememberMe = document.getElementById('rememberMe').checked;
            const alertBox = document.getElementById('alertBox');
            const submitBtn = document.getElementById('submitBtn');

            alertBox.style.display = 'none';
            submitBtn.disabled = true;
            submitBtn.textContent = 'Signing in...';

            try {{
                const res = await fetch('/api/v1/auth/login', {{
                    method: 'POST',
                    credentials: 'include',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ email, password, remember_me: rememberMe }})
                }});
                const data = await res.json();
                if (!res.ok) {{
                    throw new Error(data.detail || 'Invalid email or password.');
                }}

                // Store token in localStorage for Bearer authorization fallback
                if (data.access_token) {{
                    localStorage.setItem('zenova_token', data.access_token);
                    localStorage.setItem('zenova_user', JSON.stringify(data.user));
                }}

                alertBox.className = 'alert-banner alert-success';
                alertBox.textContent = 'Signed in successfully. Redirecting...';
                alertBox.style.display = 'block';

                const params = new URLSearchParams(window.location.search);
                let redirect = params.get('redirect') || '/app';
                if (!redirect || redirect.includes('/login') || redirect.includes('login')) {{
                    redirect = '/app';
                }}
                setTimeout(() => {{ window.location.href = redirect; }}, 300);
            }} catch (err) {{
                alertBox.className = 'alert-banner alert-danger';
                alertBox.textContent = err.message;
                alertBox.style.display = 'block';
                submitBtn.disabled = false;
                submitBtn.textContent = 'Sign In';
            }}
        }}

        // If user is already authenticated with a valid token, auto-redirect directly to app
        window.addEventListener('DOMContentLoaded', async () => {{
            const token = localStorage.getItem('zenova_token');
            if (token) {{
                try {{
                    const res = await fetch('/api/v1/auth/me', {{
                        credentials: 'include',
                        headers: {{ 'Authorization': `Bearer ${{token}}` }}
                    }});
                    if (res.ok) {{
                        const params = new URLSearchParams(window.location.search);
                        let redirect = params.get('redirect') || '/app';
                        if (!redirect || redirect.includes('/login') || redirect.includes('login')) {{
                            redirect = '/app';
                        }}
                        window.location.href = redirect;
                    }}
                }} catch (e) {{
                    // Token expired or invalid, stay on login page
                }}
            }}
        }});
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@router.get("/register", response_class=HTMLResponse)
async def register_page() -> HTMLResponse:
    """Serve the ZENOVA Registration Page."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZENOVA — Create Account</title>
    <style>{AUTH_PAGE_CSS}</style>
</head>
<body>
    <button class="theme-toggle-corner" onclick="toggleTheme()">?? Theme</button>
    <main class="auth-card" role="main">
        <div class="brand-header">
            <div class="logo-badge">ZENOVA</div>
            <h1>Create Account</h1>
            <p>Begin your personal wellbeing journey</p>
        </div>

        <div id="alertBox" class="alert-banner" role="alert"></div>

        <form id="regForm" onsubmit="handleRegister(event)">
            <div class="form-group">
                <label for="displayName">Preferred Name or Alias (Optional)</label>
                <input id="displayName" type="text" class="form-control" placeholder="How you would like ZENOVA to address you">
            </div>

            <div class="form-group">
                <label for="email">Email Address</label>
                <input id="email" type="email" class="form-control" required autocomplete="email" placeholder="name@example.com">
            </div>

            <div class="form-group">
                <label for="password">Password (Minimum 8 characters)</label>
                <div class="input-wrapper">
                    <input id="password" type="password" class="form-control" required minlength="8" autocomplete="new-password" placeholder="Choose a secure password">
                    <button type="button" class="password-toggle-btn" onclick="togglePassword('password')">Show</button>
                </div>
            </div>

            <div class="form-group">
                <label for="confirmPassword">Confirm Password</label>
                <div class="input-wrapper">
                    <input id="confirmPassword" type="password" class="form-control" required minlength="8" autocomplete="new-password" placeholder="Re-type your password">
                    <button type="button" class="password-toggle-btn" onclick="togglePassword('confirmPassword')">Show</button>
                </div>
            </div>

            <div class="form-group" style="font-size: 12px; margin-top: 10px;">
                <label style="display: flex; align-items: flex-start; gap: 8px; cursor: pointer; font-weight: normal; color: var(--text-muted);">
                    <input id="termsAgree" type="checkbox" required style="margin-top: 2px;">
                    <span>I acknowledge that ZENOVA is an AI wellbeing support tool and does not provide psychiatric diagnoses or medical treatments.</span>
                </label>
            </div>

            <button id="submitBtn" type="submit" class="btn-submit">Create Account</button>
        </form>

        <div class="auth-links">
            <div>Already have an account? <a href="/login">Sign in</a></div>
        </div>

        <div class="disclaimer-box">
            Your personal data is encrypted and kept private under strict GDPR and data ownership policies.
        </div>
    </main>

    <script>
        function toggleTheme() {{
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            localStorage.setItem('zenova-theme', next);
        }}
        const savedTheme = localStorage.getItem('zenova-theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);

        function togglePassword(id) {{
            const input = document.getElementById(id);
            const btn = event.target;
            if (input.type === 'password') {{
                input.type = 'text';
                btn.textContent = 'Hide';
            }} else {{
                input.type = 'password';
                btn.textContent = 'Show';
            }}
        }}

        async function handleRegister(e) {{
            e.preventDefault();
            const displayName = document.getElementById('displayName').value.trim();
            const email = document.getElementById('email').value.trim();
            const password = document.getElementById('password').value;
            const confirmPassword = document.getElementById('confirmPassword').value;
            const alertBox = document.getElementById('alertBox');
            const submitBtn = document.getElementById('submitBtn');

            if (password !== confirmPassword) {{
                alertBox.className = 'alert-banner alert-danger';
                alertBox.textContent = 'Passwords do not match.';
                alertBox.style.display = 'block';
                return;
            }}

            alertBox.style.display = 'none';
            submitBtn.disabled = true;
            submitBtn.textContent = 'Creating account...';

            try {{
                const res = await fetch('/api/v1/auth/register', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        email: email,
                        password: password,
                        confirm_password: confirmPassword,
                        display_name: displayName || null
                    }})
                }});
                const data = await res.json();
                if (!res.ok) {{
                    throw new Error(data.detail || 'Registration failed. Please check your inputs.');
                }}

                alertBox.className = 'alert-banner alert-success';
                alertBox.textContent = 'Account created successfully! Redirecting to sign in...';
                alertBox.style.display = 'block';

                setTimeout(() => {{ window.location.href = '/login?registered=1'; }}, 1000);
            }} catch (err) {{
                alertBox.className = 'alert-banner alert-danger';
                alertBox.textContent = err.message;
                alertBox.style.display = 'block';
                submitBtn.disabled = false;
                submitBtn.textContent = 'Create Account';
            }}
        }}
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@router.get("/forgot-password", response_class=HTMLResponse)
async def forgot_password_page() -> HTMLResponse:
    """Serve the Forgot Password Page."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZENOVA — Reset Password</title>
    <style>{AUTH_PAGE_CSS}</style>
</head>
<body>
    <main class="auth-card" role="main">
        <div class="brand-header">
            <div class="logo-badge">ZENOVA</div>
            <h1>Password Reset</h1>
            <p>Enter your email to receive recovery instructions</p>
        </div>

        <div id="alertBox" class="alert-banner" role="alert"></div>

        <form id="forgotForm" onsubmit="handleForgot(event)">
            <div class="form-group">
                <label for="email">Email Address</label>
                <input id="email" type="email" class="form-control" required placeholder="name@example.com">
            </div>

            <button id="submitBtn" type="submit" class="btn-submit">Send Reset Link</button>
        </form>

        <div class="auth-links">
            <div>Remembered your password? <a href="/login">Return to Sign In</a></div>
        </div>
    </main>

    <script>
        async function handleForgot(e) {{
            e.preventDefault();
            const email = document.getElementById('email').value.trim();
            const alertBox = document.getElementById('alertBox');
            const submitBtn = document.getElementById('submitBtn');

            submitBtn.disabled = true;
            submitBtn.textContent = 'Sending...';

            try {{
                const res = await fetch('/api/v1/auth/forgot-password', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ email }})
                }});
                const data = await res.json();
                alertBox.className = 'alert-banner alert-success';
                alertBox.textContent = data.message;
                alertBox.style.display = 'block';
                document.getElementById('forgotForm').reset();
            }} catch (err) {{
                alertBox.className = 'alert-banner alert-danger';
                alertBox.textContent = 'Unable to submit request. Please try again.';
                alertBox.style.display = 'block';
            }} finally {{
                submitBtn.disabled = false;
                submitBtn.textContent = 'Send Reset Link';
            }}
        }}
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@router.get("/reset-password", response_class=HTMLResponse)
async def reset_password_page() -> HTMLResponse:
    """Serve the Set New Password Page."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZENOVA — Set New Password</title>
    <style>{AUTH_PAGE_CSS}</style>
</head>
<body>
    <main class="auth-card" role="main">
        <div class="brand-header">
            <div class="logo-badge">ZENOVA</div>
            <h1>Set New Password</h1>
            <p>Create a fresh, secure password for your account</p>
        </div>

        <div id="alertBox" class="alert-banner" role="alert"></div>

        <form id="resetForm" onsubmit="handleReset(event)">
            <input type="hidden" id="token">

            <div class="form-group">
                <label for="newPassword">New Password (Minimum 8 characters)</label>
                <div class="input-wrapper">
                    <input id="newPassword" type="password" class="form-control" required minlength="8" placeholder="Enter new password">
                </div>
            </div>

            <div class="form-group">
                <label for="confirmPassword">Confirm New Password</label>
                <div class="input-wrapper">
                    <input id="confirmPassword" type="password" class="form-control" required minlength="8" placeholder="Re-type new password">
                </div>
            </div>

            <button id="submitBtn" type="submit" class="btn-submit">Reset Password</button>
        </form>

        <div class="auth-links">
            <div><a href="/login">Back to Sign In</a></div>
        </div>
    </main>

    <script>
        const urlParams = new URLSearchParams(window.location.search);
        const token = urlParams.get('token');
        if (!token) {{
            const alertBox = document.getElementById('alertBox');
            alertBox.className = 'alert-banner alert-danger';
            alertBox.textContent = 'Invalid or missing password reset token. Please request a new link.';
            alertBox.style.display = 'block';
            document.getElementById('submitBtn').disabled = true;
        }} else {{
            document.getElementById('token').value = token;
        }}

        async function handleReset(e) {{
            e.preventDefault();
            const tokenVal = document.getElementById('token').value;
            const newPassword = document.getElementById('newPassword').value;
            const confirmPassword = document.getElementById('confirmPassword').value;
            const alertBox = document.getElementById('alertBox');
            const submitBtn = document.getElementById('submitBtn');

            if (newPassword !== confirmPassword) {{
                alertBox.className = 'alert-banner alert-danger';
                alertBox.textContent = 'Passwords do not match.';
                alertBox.style.display = 'block';
                return;
            }}

            submitBtn.disabled = true;
            submitBtn.textContent = 'Resetting...';

            try {{
                const res = await fetch('/api/v1/auth/reset-password', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        token: tokenVal,
                        new_password: newPassword,
                        confirm_new_password: confirmPassword
                    }})
                }});
                const data = await res.json();
                if (!res.ok) {{
                    throw new Error(data.detail || 'Password reset failed.');
                }}
                alertBox.className = 'alert-banner alert-success';
                alertBox.textContent = data.message;
                alertBox.style.display = 'block';
                setTimeout(() => {{ window.location.href = '/login?reset=1'; }}, 1200);
            }} catch (err) {{
                alertBox.className = 'alert-banner alert-danger';
                alertBox.textContent = err.message;
                alertBox.style.display = 'block';
                submitBtn.disabled = false;
                submitBtn.textContent = 'Reset Password';
            }}
        }}
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@router.get("/verify-email", response_class=HTMLResponse)
async def verify_email_page() -> HTMLResponse:
    """Serve the Email Verification Landing Page."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZENOVA — Verify Email</title>
    <style>{AUTH_PAGE_CSS}</style>
</head>
<body>
    <main class="auth-card" role="main">
        <div class="brand-header">
            <div class="logo-badge">ZENOVA</div>
            <h1>Email Verification</h1>
            <p id="subheading">Confirming your account...</p>
        </div>

        <div id="alertBox" class="alert-banner" role="alert" style="display: block;">Checking verification token...</div>

        <div class="auth-links" style="margin-top: 24px;">
            <div><a href="/login">Proceed to Sign In</a></div>
        </div>
    </main>

    <script>
        const urlParams = new URLSearchParams(window.location.search);
        const token = urlParams.get('token');
        const alertBox = document.getElementById('alertBox');
        const subheading = document.getElementById('subheading');

        if (!token) {{
            alertBox.className = 'alert-banner alert-danger';
            alertBox.textContent = 'No verification token provided. Please use the link sent to your email.';
            subheading.textContent = 'Verification Failed';
        }} else {{
            fetch('/api/v1/auth/verify-email', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ token }})
            }})
            .then(res => res.json().then(data => ({{ ok: res.ok, data }})))
            .then(({{ ok, data }}) => {{
                if (ok) {{
                    alertBox.className = 'alert-banner alert-success';
                    alertBox.textContent = data.message || 'Email verified successfully!';
                    subheading.textContent = 'Verification Complete';
                    setTimeout(() => {{ window.location.href = '/login?verified=1'; }}, 1500);
                }} else {{
                    alertBox.className = 'alert-banner alert-danger';
                    alertBox.textContent = data.detail || 'Verification token is invalid or has expired.';
                    subheading.textContent = 'Verification Unsuccessful';
                }}
            }})
            .catch(err => {{
                alertBox.className = 'alert-banner alert-danger';
                alertBox.textContent = 'Network error while verifying email.';
                subheading.textContent = 'Error';
            }});
        }}
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)
