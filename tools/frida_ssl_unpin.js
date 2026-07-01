/**
 * 通用 Android HTTPS 证书校验绕过（抓包用）。
 * 用法见 tools/frida_capture_help.py 或文档说明。
 */
Java.perform(function () {
  console.log("[*] SSL unpinning script loaded");

  // Trust all certs (Android 7+ TrustManagerImpl)
  try {
    var TrustManagerImpl = Java.use("com.android.org.conscrypt.TrustManagerImpl");
    TrustManagerImpl.verifyChain.implementation = function (
      untrustedChain,
      trustAnchorChain,
      host,
      clientAuth,
      ocspData,
      tlsSctData
    ) {
      console.log("[+] TrustManagerImpl.verifyChain bypass: " + host);
      return untrustedChain;
    };
  } catch (e) {
    console.log("[-] TrustManagerImpl: " + e);
  }

  // SSLContext.init → empty TrustManager
  try {
    var X509TrustManager = Java.use("javax.net.ssl.X509TrustManager");
    var SSLContext = Java.use("javax.net.ssl.SSLContext");
    var TrustManager = Java.registerClass({
      name: "com.endlesswinter.TrustAllManager",
      implements: [X509TrustManager],
      methods: {
        checkClientTrusted: function (chain, authType) {},
        checkServerTrusted: function (chain, authType) {},
        getAcceptedIssuers: function () {
          return [];
        },
      },
    });
    var TrustManagers = [TrustManager.$new()];
    var SSLContextInit = SSLContext.init.overload(
      "[Ljavax.net.ssl.KeyManager;",
      "[Ljavax.net.ssl.TrustManager;",
      "java.security.SecureRandom"
    );
    SSLContextInit.implementation = function (km, tm, sr) {
      console.log("[+] SSLContext.init bypass");
      SSLContextInit.call(this, km, TrustManagers, sr);
    };
  } catch (e) {
    console.log("[-] SSLContext: " + e);
  }

  // OkHttp3 CertificatePinner
  try {
    var CertificatePinner = Java.use("okhttp3.CertificatePinner");
    CertificatePinner.check.overload("java.lang.String", "java.util.List").implementation =
      function (hostname, peerCertificates) {
        console.log("[+] OkHttp3 CertificatePinner bypass: " + hostname);
      };
  } catch (e) {
    console.log("[-] OkHttp3 CertificatePinner: " + e);
  }

  try {
    var CertificatePinner2 = Java.use("okhttp3.CertificatePinner");
    CertificatePinner2.check.overload("java.lang.String", "[Ljava.security.cert.Certificate;")
      .implementation = function (hostname, certs) {
        console.log("[+] OkHttp3 CertificatePinner (certs) bypass: " + hostname);
      };
  } catch (e) {
    console.log("[-] OkHttp3 CertificatePinner certs: " + e);
  }

  // HttpsURLConnection
  try {
    var HttpsURLConnection = Java.use("javax.net.ssl.HttpsURLConnection");
    HttpsURLConnection.setDefaultHostnameVerifier.implementation = function (verifier) {
      console.log("[+] HttpsURLConnection.setDefaultHostnameVerifier bypass");
      return null;
    };
  } catch (e) {
    console.log("[-] HttpsURLConnection: " + e);
  }

  console.log("[*] SSL unpinning hooks installed");
});
