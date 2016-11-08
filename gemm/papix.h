#pragma once

#ifdef __INTELLISENSE__
#define USE_PAPI
#endif

#ifdef USE_PAPI

#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include <papi.h>

#define PAPI_CHECK(call) \
    do { \
        int ret__ = call; \
        if (ret__ != PAPI_OK) { \
            const char * err = PAPI_strerror(ret__); \
            std::printf("PAPI error: %s (%s:%d)\n", \
                err, __FILE__, __LINE__); \
            std::exit(1); \
        } \
    } while (0)

class papix {
    enum : int {
        max_events = 5,
    };

    struct prof_entry {
        std::string name;
        size_t count;
        long long total[max_events];
        long long min[max_events];
        long long max[max_events];

        prof_entry() : name(), count(0)
        {
            for (int i = 0; i < max_events; i++) {
                total[i] = 0;
                min[i] = std::numeric_limits<long long>::max();
                max[i] = std::numeric_limits<long long>::min();
            }
        }
    };

    bool profiling_;
    std::vector<int> events_;
    std::vector<std::string> event_strs_;
    std::map<std::string, prof_entry> prof_entries_;

public:
    papix()
        : profiling_(false)
        , events_()
        , event_strs_()
        , prof_entries_()
    {
        auto events_str = getenv("PAPIX_EVENTS");
        auto pair = parse_event_list(events_str);
        events_ = pair.first;
        event_strs_ = pair.second;
    }

    ~papix()
    {
    }

    template <class F>
    void measure(const char * name, F f)
    {
        if (profiling_) return;

        auto n_events = events_.size();
        PAPI_CHECK(PAPI_start_counters(
            const_cast<int *>(events_.data()), n_events));

        f();

        long long values[256];
        PAPI_CHECK(PAPI_stop_counters(values, n_events));

        submit(name, values, n_events);

#if 0
        for (int i = 0; i < n_events; i++) {
            char name[PAPI_MAX_STR_LEN];
            PAPI_CHECK(PAPI_event_code_to_name(events_[i], name));
            printf("%-16s : %10lld\n", name, values[i]);
        }
#endif
    }

    void print_results()
    {
        if (prof_entries_.size() == 0) return;

        std::printf("\n");
        std::printf("==== PAPI Results ====\n");
        std::printf("%-20s %-12s %7s %11s %10s %10s %10s\n",
            "func", "counter", "count", "total", "avg", "min", "max");

        for (auto& pair : prof_entries_) {
            auto& name = pair.first;
            auto& e = pair.second;

            for (size_t i = 0; i < events_.size(); i++) {
                std::printf(
                    "%-20s %-12s %7zu %11lld %10lld %10lld %10lld\n",
                    name.c_str(), event_strs_[i].c_str(), e.count,
                    e.total[i], e.total[i] / e.count,
                    e.min[i], e.max[i]);
            }
        }
    }

private:
    std::map<std::string, int> make_name2code()
    {
        std::map<std::string, int> map;

#define ADD(name) map[#name] = name

        ADD(PAPI_L1_DCM); ADD(PAPI_L1_ICM); ADD(PAPI_L2_DCM); ADD(PAPI_L2_ICM);
        ADD(PAPI_L3_DCM); ADD(PAPI_L3_ICM); ADD(PAPI_L1_TCM); ADD(PAPI_L2_TCM);
        ADD(PAPI_L3_TCM); ADD(PAPI_CA_SNP); ADD(PAPI_CA_SHR); ADD(PAPI_CA_CLN);
        ADD(PAPI_CA_INV); ADD(PAPI_CA_ITV); ADD(PAPI_L3_LDM); ADD(PAPI_L3_STM);
        ADD(PAPI_BRU_IDL); ADD(PAPI_FXU_IDL); ADD(PAPI_FPU_IDL);
        ADD(PAPI_LSU_IDL); ADD(PAPI_TLB_DM); ADD(PAPI_TLB_IM); ADD(PAPI_TLB_TL);
        ADD(PAPI_L1_LDM); ADD(PAPI_L1_STM); ADD(PAPI_L2_LDM); ADD(PAPI_L2_STM);
        ADD(PAPI_BTAC_M); ADD(PAPI_PRF_DM); ADD(PAPI_L3_DCH); ADD(PAPI_TLB_SD);
        ADD(PAPI_CSR_FAL); ADD(PAPI_CSR_SUC); ADD(PAPI_CSR_TOT);
        ADD(PAPI_MEM_SCY); ADD(PAPI_MEM_RCY); ADD(PAPI_MEM_WCY);
        ADD(PAPI_STL_ICY); ADD(PAPI_FUL_ICY); ADD(PAPI_STL_CCY);
        ADD(PAPI_FUL_CCY); ADD(PAPI_HW_INT); ADD(PAPI_BR_UCN); ADD(PAPI_BR_CN);
        ADD(PAPI_BR_TKN); ADD(PAPI_BR_NTK); ADD(PAPI_BR_MSP); ADD(PAPI_BR_PRC);
        ADD(PAPI_FMA_INS); ADD(PAPI_TOT_IIS); ADD(PAPI_TOT_INS);
        ADD(PAPI_INT_INS); ADD(PAPI_FP_INS); ADD(PAPI_LD_INS); ADD(PAPI_SR_INS);
        ADD(PAPI_BR_INS); ADD(PAPI_VEC_INS); ADD(PAPI_RES_STL);
        ADD(PAPI_FP_STAL); ADD(PAPI_TOT_CYC); ADD(PAPI_LST_INS);
        ADD(PAPI_SYC_INS); ADD(PAPI_L1_DCH); ADD(PAPI_L2_DCH); ADD(PAPI_L1_DCA);
        ADD(PAPI_L2_DCA); ADD(PAPI_L3_DCA); ADD(PAPI_L1_DCR); ADD(PAPI_L2_DCR);
        ADD(PAPI_L3_DCR); ADD(PAPI_L1_DCW); ADD(PAPI_L2_DCW); ADD(PAPI_L3_DCW);
        ADD(PAPI_L1_ICH); ADD(PAPI_L2_ICH); ADD(PAPI_L3_ICH); ADD(PAPI_L1_ICA);
        ADD(PAPI_L2_ICA); ADD(PAPI_L3_ICA); ADD(PAPI_L1_ICR); ADD(PAPI_L2_ICR);
        ADD(PAPI_L3_ICR); ADD(PAPI_L1_ICW); ADD(PAPI_L2_ICW); ADD(PAPI_L3_ICW);
        ADD(PAPI_L1_TCH); ADD(PAPI_L2_TCH); ADD(PAPI_L3_TCH); ADD(PAPI_L1_TCA);
        ADD(PAPI_L2_TCA); ADD(PAPI_L3_TCA); ADD(PAPI_L1_TCR); ADD(PAPI_L2_TCR);
        ADD(PAPI_L3_TCR); ADD(PAPI_L1_TCW); ADD(PAPI_L2_TCW); ADD(PAPI_L3_TCW);
        ADD(PAPI_FML_INS); ADD(PAPI_FAD_INS); ADD(PAPI_FDV_INS);
        ADD(PAPI_FSQ_INS); ADD(PAPI_FNV_INS); ADD(PAPI_FP_OPS); ADD(PAPI_SP_OPS);
        ADD(PAPI_DP_OPS); ADD(PAPI_VEC_SP); ADD(PAPI_VEC_DP); ADD(PAPI_REF_CYC);

#undef ADD

        return map;
    }

    std::pair<std::vector<int>, std::vector<std::string>>
        parse_event_list(const char * events_str)
    {
        std::vector<int> event_list;
        std::vector<std::string> event_str_list;

        auto name2code = make_name2code();

        if (events_str == nullptr)
            return std::make_pair(event_list, event_str_list);

        std::string name;
        std::stringstream ss(events_str);
        while (std::getline(ss, name, ',')) {
            std::string evname = "PAPI_";
            evname += name;
#if 0
            int code;
            PAPI_CHECK(PAPI_event_name_to_code(&evname[0], &code));
#else
            if (name2code.find(evname) != name2code.end()) {
                int code = name2code[evname];
                event_list.push_back(code);
                event_str_list.push_back(evname);
            }
#endif
        }

        auto new_size = std::min<int>(max_events, event_list.size());
        event_list.resize(new_size);
        event_str_list.resize(new_size);

        return std::make_pair(event_list, event_str_list);
    }

    void submit(const char * name, long long values[], int n_values)
    {
        std::string sname = name;
        auto& e = prof_entries_[sname];
        e.name = sname;
        e.count += 1;

        for (int i = 0; i < n_values; i++) {
            e.total[i] += values[i];
            e.min[i] = std::min(e.min[i], values[i]);
            e.max[i] = std::max(e.max[i], values[i]);
        }
    }
};

#else // USE_PAPI

class papix {
public:
    papix() {}
    ~papix() {}
    void print_results() {}

    template <class F>
    void measure(const char * name, F f) { f(); }
};

#endif
