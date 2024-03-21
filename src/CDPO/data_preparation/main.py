import lib.file_func as file_func
import data_dpo_ultrafeedback as dpo_fb
import data_dpo_ultrasafety as dpo_sf

DPO_FEEDBACK = {
    "cfg": "../../../scripts/dpo_ultrafeedback_cfg.json",
}

DPO_SAFETY = {
    "cfg": "../../../scripts/dpo_ultrasafety_cfg.json",
}

if __name__ == "__main__":
    fb_data = file_func.readJsonFile(DPO_FEEDBACK["cfg"])
    sf_data = file_func.readJsonFile(DPO_SAFETY["cfg"])

    dpo_fb.Start(
        fb_data["input"], 
        fb_data["output"],
        fb_data["has_harmless"],
        fb_data["random_cfg"],
    )
    dpo_sf.Start(
        sf_data["input"], 
        sf_data["output"],
        sf_data["has_harmless"],
        sf_data["random_cfg"],
    )

    ##TODO:步骤自己补


