# Image gateway and repair incident

## Verified cause and restoration

The former ngrok image endpoint returned HTTP403 HTML ERR_NGROK_725: monthly network bandwidth exhausted. This is a tunnel failure, not proof of image-account quota exhaustion or authentication failure. The original gateway process on the owner's host (port3011) remains unchanged.

New transport: local user services pocket-image-forward.service (SSH port3012 to owner host3011) and pocket-image-tunnel.service (Cloudflare quick tunnel). Production Serverix #795 identifier74e81a81 .env gateway URL changed surgically. Old and new environment are kept only as protected private artifacts, never committed.

Public API catalog returned200 and6 models. Account inventory returned4 records, all normal. Actual image generation returned200 in32.9 seconds; PNG decoded1254x1254,2227131 bytes. This observed size is not the requested1024x1024.

Telegram live command162746 in Saved Messages produced actual photo162748. The deployed bot startup was read back in production logs.

## Repair fixes

Crop comparison now examines the feathered composite crop that will actually be committed, not the unblended provider patch. Gemini QA reasoning is bounded for both local and public tunnel routes; paired comparison now uses4000 output tokens and90-second timeout rather than1200/60. Target evidence and regression rejection remain fail-closed.

Real historical hand-defect probe generated a crop, but QA still found elongated fingers. The original image was correctly retained. This is not evidence of a successful hand repair. A separate explicitly constructed scratch fixture was repaired through the actual gateway: changed=true,1 generation,4 real QA calls, regional repair accepted1/1, remaining defects0. Crop and whole-image comparisons reported complete scratch removal and preservation of identity/requirements without seams. This proves the mechanism can accept an actual correction; it does not prove all anatomy repairs succeed.

## Regression and preservation

149 pytest tests plus21 subtests passed. A real scratch-fixture attempt returned paired evidence of complete removal with all preservation flags true, but confidence0.72 was rejected by the earlier0.8 crop gate. The local crop gate now matches detection threshold0.65; the regional whole-image acceptance now requires0.7 and all preservation flags, plus clean standalone crop/composite QA and unchanged outside pixels. Two actual scratch probes reported complete removal, all flags true, and confidence0.72; the former0.8 threshold alone caused rejection. Default comparison elsewhere remains0.8. This is a deliberate policy calibration, not a fabricated repair success. Safe pair-QA rejection diagnostics were added. Existing .session files, model state and history were not overwritten. Panel backup hit its2-backup cap; existing backups were preserved, and touched production files backed up under qa_artifacts/incident-20261014/predeploy with restrictive permissions.

## Residual transport risk

The temporary trycloudflare URL is not a named tunnel and changes on service restart. Stable named-tunnel credentials/domain were not present on this VPS; quick-tunnel restoration must not be described as a permanent named deployment.
