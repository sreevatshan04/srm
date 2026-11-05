import { getApp, getApps, initializeApp } from 'firebase/app';
import { getAnalytics, isSupported } from 'firebase/analytics';
import { getAuth } from 'firebase/auth';

// Firebase web configuration is public client configuration. Security is enforced
// by Firebase Authentication and backend-verified ID tokens, not by hiding this key.
const firebaseConfig = {
  apiKey: "AIzaSyCg7QQmcbCt8X9Wy_fOh05bAPQnLh_UKow",
  authDomain: "messmate-production.firebaseapp.com",
  projectId: "messmate-production",
  storageBucket: "messmate-production.firebasestorage.app",
  messagingSenderId: "279941774828",
  appId: "1:279941774828:web:a6a43b68f44a35668194c9",
  measurementId: "G-V4B9P30LH0"
};

export const firebaseApp = getApps().length ? getApp() : initializeApp(firebaseConfig);
export const auth = getAuth(firebaseApp);

// Analytics is optional and is only initialized in browsers where Firebase supports it.
export const analyticsReady = isSupported()
  .then(supported => (supported ? getAnalytics(firebaseApp) : null))
  .catch(() => null);
